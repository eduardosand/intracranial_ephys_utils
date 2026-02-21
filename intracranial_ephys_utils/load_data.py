from neo.rawio import NeuralynxRawIO
import numpy as np
import os
from scipy.interpolate import CubicSpline
import warnings
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

def get_file_info(directory: Path, start: str, file_extension: str) -> Path:
    """
    Look in directory for files that start and end with something. Raise error if more than one match is found
    Useful for neuralynx files where they may be multiple session, and file identifiers.

    Args:
        directory (Path):  Path to the directory we want to look through.
        start (string):  start prefix of desired file
        file_extension (string):  end suffix of desired file

    Returns:
        File_path: first file path that matches

    Examples:
        >>> get_file_info(Path(os.getcwd()), 'L', '.txt') # grab the license from this repo
        PosixPath('/home/runner/work/intracranial_ephys_utils/intracranial_ephys_utils/LICENSE.txt')
    """
    files_list = os.listdir(directory)
    files = [file_path for file_path in files_list if file_path.endswith(file_extension) and
             file_path.startswith(start)]
    if len(files) > 1:
        print(f'Found files: {files}')
        warnings.warn(f"More than one file matching {start}*{file_extension}, consider a more specific function call.")
        file_path = directory / files[0]
        return file_path
        #raise ValueError(f"More than one file matching {start}*{file_extension}, consider a more specific function call.")
    file_path = directory / files[0]
    return file_path


def read_file(file_path: Path) -> NeuralynxRawIO:
    """
    Lazy reader of specific Neuralynx files. Will probably be removed eventually

    Args:
        file_path (Path):  Absolute path to a neuralynx .ncs file.

    Returns:
        reader_object: neo reader object for neuralynx data
    """
    stems = os.path.split(file_path)
    reader = NeuralynxRawIO(dirname=stems[0], include_filenames=stems[1])
    return reader


def get_event_times(neuralynx_data_directory: Path, extension: Optional[str] = None) -> tuple[list, list, float, list]:
    """
    Looks at just the events file for a Neuralynx data directory to get timestamps(default is seconds) and labels
    for recording events. This is useful if researcher of clinician manually annotated events while recording. Keep in
    mind however, that the events file is sometimes not in accordance with the data itself, due to a referencing issue.

    Args:
        neuralynx_data_directory (Path): Directory of Neuralynx data, that stores at least one .nev file
        extension: optional(default None). If there are multiple .nev file, can specify the desired .nev by the end of the filename

    Returns:
        event_times : machine time in microseconds. To convert this to seconds relative to recording start, convert to seconds and subtract from global_start
        event_labels : Whatever the annotation was for each event.
        global_start : Machine code time beginning of recording(seconds)
        event_files : File names for events file (useful in case there are two events files)
    """
    # Obtained in seconds and assumes that the start of the file (not necessarily the task) is 0.

    all_files = os.listdir(neuralynx_data_directory)

    # Find the events file, note just the filename, not the full path
    if extension is None:
        events_files = [file_path for file_path in all_files if file_path.startswith('Events') or file_path.endswith('.nev')]
        # full_events_filepath = get_file_info(neuralynx_data_directory, 'Events', '.nev')
        # events_file = full_events_filepath.name
    else:
        events_files = [file_path for file_path in all_files if (file_path.startswith('Events') and
                                                                file_path.endswith(extension))]
        # full_events_filepath = get_file_info(neuralynx_data_directory, 'Events', extension)
        # events_file = full_events_filepath.name

    if len(events_files) == 1:
        # with one events file, read in the photodiode file
        events_file = events_files[0]
        event_reader = read_file(neuralynx_data_directory / events_file)
        event_reader.parse_header()

        ph_file = events_file.replace(".nev", ".ncs")
        ph_file = ph_file.replace("Events", "photo1")
        if os.path.exists(neuralynx_data_directory / ph_file):
            ph_reader = read_file(neuralynx_data_directory / ph_file)
        else:
            # pick other file by random
            file_list = os.listdir(neuralynx_data_directory)
            better_file_list = [file for file in file_list if file.endswith('.ncs')]
            ph_reader = read_file(neuralynx_data_directory / better_file_list[0])
            warnings.warn(f"No photodiode file found. Using {better_file_list[0]}")
        try:
            ph_reader.parse_header()
            global_start = ph_reader.global_t_start
        except OSError:
            warnings.warn("OSError. Inquire further. Previous errors were from a header missing information.")
        # global_start_event_reader = event_reader.global_t_start
        try:
            event_times, _, event_labels = event_reader.get_event_timestamps()
            if 'global_start' not in locals():
                global_start = event_times[0]
                warnings.warn("No files to choose from. Is there neuralynx data in this folder?")
        except IndexError:
            warnings.warn("No events found")
            event_times, event_labels = [], []
            ph_path = get_file_info(neuralynx_data_directory, "photo", ".ncs")
            ph_reader = read_file(ph_path)
            ph_reader.parse_header()
            global_start = ph_reader.global_t_start
        events_files = [events_file]
    elif len(events_files) == 0:
        warnings.warn("No events file found, Using Photodiode file to get global machine time start")
        event_times, event_labels = [], []
        ph_path = get_file_info(neuralynx_data_directory, "photo", ".ncs")
        ph_reader = read_file(ph_path)
        ph_reader.parse_header()
        global_start = ph_reader.global_t_start
        events_file = None
    else:
        print(f'Found files: {events_files}')
    return event_times, event_labels, global_start, events_files


def missing_samples_check(file_path: Path) -> tuple[list, list, list]:
    """
    This script checks for missing samples in neuralynx files.

    Args:
        file_path: Path object where data file is.

    Returns:
        skipped_samples: integer number of skipped samples
        t_starts: The start times for each segment
        seg_sizes: The lengths of each Neo segment
    """
    file_reader = read_file(file_path)
    file_reader.parse_header()
    n_segments = file_reader._nb_segment
    t_starts = []
    seg_sizes = []
    sampling_rate = file_reader.get_signal_sampling_rate()
    skipped_samples = []
    diffs = []
    # Start of the last segment + size of last segment - start of the task according to event
    # ph_signal_estimate = file_reader.get_signal_t_start(block_index=0, seg_index=n_segments - 1) \
    #                      * sampling_rate + \
    #                      float(file_reader.get_signal_size(block_index=0, seg_index=n_segments - 1))
    # ph_signal = np.zeros((int(ph_signal_estimate), 1))
    for i in range(n_segments):
        t_start = file_reader.get_signal_t_start(block_index=0, seg_index=i)
        seg_size = file_reader.get_signal_size(block_index=0, seg_index=i)
        # ph_signal_segment = file_reader.get_analogsignal_chunk(seg_index=i)
        # start_index = int(t_start * sampling_rate)
        # ph_signal[start_index:start_index + seg_size] = ph_signal_segment
        if i > 0:
            # This part of the script looks for missing samples
            t_end = float(seg_sizes[i - 1] / sampling_rate) + t_starts[- 1]
            diff = abs(t_start - t_end) * sampling_rate
            skipped_samples.append(round(diff))
            diffs.append(abs(t_start - t_end))
        t_starts.append(t_start)
        seg_sizes.append(seg_size)
        print(diffs)
    return skipped_samples, t_starts, seg_sizes


def read_task_ncs(folder_name: Path, file: str, task: Optional[str]=None, events_file: Optional[Path]=None,
                  interp_type: Optional[str]='linear') -> tuple[np.array, float, np.array, np.array]:
    """
    Read neuralynx data into an array, with sampling rate, and start time of the task.
    To deal with discontinuities and dropped samples, we take a pragmatic approach. We assume continuous sampling, and
    if there are inconsistencies between the number of samples in segments and the array itself, we fill in samples by
    interpolating.
    Ideally this spits out neuralynx data in the form of an array, with the sampling rate, and the start time of the task

    Args:
        folder_name: Path object that tells the path of the structure
        file: filename we want to read currently
        task: (string, optional) that matches the event label in the actual events file. Ideally it matches the name of the task
        events_file: (path, optional) needed if task argument is provided, this used to be the .nev file, but I found it useless so now it's a csv file that I generate via the scripts in manual_process
        interp_type: (string, optional) what type of interpolation to do in the case of missing data, choices are
    linear or cubic, default is linear

    Returns:
        ncs_signal: ndarray - signal in the file in np array format
        sampling_rate: sampling rate for signal
        interp: ndarray - same size as ncs_signal and timestamps, tells you whether data was interpolated in that point, useful if finding weird things in data
        timestamps: an array that gives the timestamps from the ncs file using the start and stop task segments,
    this is in seconds, from the start of the .ncs file recording
    """

    file_path = folder_name / file
    ncs_reader = read_file(file_path)
    ncs_reader.parse_header()
    n_segments = ncs_reader._nb_segment
    sampling_rate = ncs_reader.get_signal_sampling_rate()

    # This loop is to get around files that have weird events files, or task wasn't in the annotation
    if task is not None:
        if events_file is None:
            raise ValueError("Need to provide event file")
        labels_file = pd.read_csv(events_file)
        task_label = labels_file[labels_file.label == f"{task} duration"]
        task_start = task_label['time'].iloc[0].astype(float)  # seconds from start of file
        task_end = task_start + task_label['duration'].iloc[0].astype(float)

        # print(float(ncs_reader.get_signal_size(block_index=0, seg_index=n_segments - 1))/sampling_rate)

        # The following block looks for the time of the start and end of the task we care about
        task_start_segment_index = None
        task_end_segment_index = None
        task_start_search = True
        for i in range(n_segments):
            time_segment_start = ncs_reader.get_signal_t_start(block_index=0, seg_index=i)
            if time_segment_start < task_start:
                continue
            elif (time_segment_start >= task_start) and (time_segment_start < task_end):
                # The first time this is run, the task_start_search bit flips
                if task_start_search:
                    task_start_search = False
                    # We take the index before because time_segment_start may not overlap with the start of the segment
                    # and this is looking from below, so overlap is with previous segment
                    task_start_segment_index = max(i - 1, 0)
            elif time_segment_start > task_end:
                # The end isn't as important if we overshoot
                task_end_segment_index = i-1
                break
            if i == n_segments-1:
                task_end_segment_index = n_segments-1
    else:
        task_start_segment_index = 0
        task_end_segment_index = n_segments-1
        task_start = 0
        task_end = round(ncs_reader.segment_t_stop(block_index=0, seg_index=task_end_segment_index), 4)

    # I believe this is in number of seconds till start(if theoretically correct), the problem is that the sampling
    # rate is an average given to us by neuralynx
    task_start_segment_time = round(ncs_reader.get_signal_t_start(block_index=0,
                                                                  seg_index=task_start_segment_index), 4) # seconds
    # to be precise ignore a certain number of samples from task_start_segment_time
    # we should really only ever be 4 decimal place precise for up to 32K
    task_start_segment_diff = int(round(task_start - task_start_segment_time, 4) * sampling_rate)  # samples

    # to be precise ignore that last n samples past task_end
    task_end_segment_time = round(ncs_reader.segment_t_stop(block_index=0, seg_index=task_end_segment_index), 4)
    task_end_segment_diff = int(round(task_end_segment_time-task_end, 4) * sampling_rate)

    array_size = round(task_end-task_start, 4) * sampling_rate
    timestamps = np.linspace(task_start, task_end, int(array_size))
    interp = np.zeros((int(array_size), ))
    ncs_signal = np.zeros((int(array_size), ))
    for i in range(task_start_segment_index, task_end_segment_index+1):
        # First stop. Get the time_segment_start and t_end for each segment.
        time_segment_start = ncs_reader.get_signal_t_start(block_index=0, seg_index=i)
        seg_size = ncs_reader.get_signal_size(block_index=0, seg_index=i)  # samples
        signal_segment = ncs_reader.get_analogsignal_chunk(seg_index=i)
        if i == task_start_segment_index:
            total_segment = ncs_reader.rescale_signal_raw_to_float(signal_segment, dtype='float32').T[0]
            start_index = 0
            # get the segment size after discounting those starting samples
            start_seg_size = int(round(seg_size - task_start_segment_diff, 4))
            ncs_signal[start_index: start_index+start_seg_size] = total_segment[task_start_segment_diff:]
        elif i == task_end_segment_index:
            total_segment = ncs_reader.rescale_signal_raw_to_float(signal_segment, dtype='float32').T[0]
            end_index = len(ncs_signal)
            # get the segment size after discounting the last samples past task_end
            end_seg_size = int(round(seg_size - task_end_segment_diff, 4))
            ncs_signal[end_index-end_seg_size:] = total_segment[:end_seg_size]
        else:
            start_index = int(round(time_segment_start - task_start, 4) *
                              sampling_rate)
            # rescale to uV
            ncs_signal[start_index:start_index+seg_size] = ncs_reader.rescale_signal_raw_to_float(signal_segment,
                                                                                                  dtype='float32').T[0]
        if i > task_start_segment_index:
            previous_segment_stop = ncs_reader.segment_t_stop(block_index=0, seg_index=i-1)
            if abs(time_segment_start-previous_segment_stop) < 1/sampling_rate:
                continue
            else:
                if abs(time_segment_start-previous_segment_stop) > 10:
                    print('There is at least 10 seconds of missing data see below')
                    print(previous_segment_stop, time_segment_start)
                    continue
                # 01/18/2024 - Consider a version of this script that doesn't interpolate, to allow for better alignment
                # to spike data where this has in fact not been done (spike sorting (Osort) doesn't care about
                # timestamping at all)
                previous_seg_signal = ncs_reader.get_analogsignal_chunk(seg_index=i-1)
                previous_seg_signal_scaled = ncs_reader.rescale_signal_raw_to_float(previous_seg_signal,
                                                                                    dtype='float32').T[0]
                previous_seg_time_start = ncs_reader.get_signal_t_start(block_index=0, seg_index=i-1)
                previous_seg_size_samples = ncs_reader.get_signal_size(block_index=0, seg_index=i-1)
                curr_seg_time_end = ncs_reader.segment_t_stop(block_index=0, seg_index=i)
                current_seg_signal_scaled = ncs_signal[start_index:start_index+seg_size]
                data_y = np.concatenate((previous_seg_signal_scaled, current_seg_signal_scaled))
                data_t = np.concatenate((np.linspace(previous_seg_time_start, previous_segment_stop,
                                                     previous_seg_size_samples), np.linspace(time_segment_start,
                                                                                             curr_seg_time_end,
                                                                                             seg_size)))

                # how many missing samples
                missing_samples_start_ind = previous_seg_size_samples-1
                missing_samples_end_ind = int((time_segment_start-previous_seg_time_start)*sampling_rate)
                # print(missing_samples_end)
                missing_samples = missing_samples_end_ind-missing_samples_start_ind
                # Define a range around the missing samples
                range_size = 100

                # Determine the sample indices for interpolation, 100 before and after the missing samples
                interpolation_start_ind = max(0, missing_samples_start_ind - range_size)
                interpolation_end_ind = min(len(data_y), missing_samples_end_ind + range_size)
                # Use only the surrounding data for interpolation
                local_data_y = data_y[interpolation_start_ind:interpolation_end_ind]
                local_data_t = data_t[interpolation_start_ind:interpolation_end_ind]

                # Create the interpolated data range over the missing sample range
                interp_data_t = np.linspace(data_t[missing_samples_start_ind], data_t[missing_samples_start_ind+1],
                                            missing_samples)

                # Select interpolation method based on the keyword
                if interp_type == 'cubic':
                    cs = CubicSpline(local_data_t, local_data_y)
                    data_x_interp = cs(interp_data_t)
                elif interp_type == 'linear':
                    data_x_interp = np.interp(interp_data_t, local_data_t, local_data_y)
                else:
                    raise NotImplementedError('Interpolation not found.')

                # data fill in
                ncs_signal[start_index-missing_samples:start_index] = data_x_interp[:missing_samples]
                interp[start_index-missing_samples:start_index] = np.ones((missing_samples_end_ind -
                                                                           missing_samples_start_ind, ))
    return ncs_signal, sampling_rate, interp, timestamps
