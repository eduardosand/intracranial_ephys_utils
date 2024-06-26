#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:10:02 2023

@author: sandoval
"""

# from neo.io import NeuralynxIO
from neo.rawio import NeuralynxRawIO
import numpy as np
import os
from scipy.interpolate import CubicSpline
import warnings
import pandas as pd

def get_file_info(directory, start, file_extension):
    """
    Look in the directory for files that start and end with something
    :param directory:
    :param start:
    :param file_extension:
    :return:
    """
    files_list = os.listdir(directory)
    files = [file_path for file_path in files_list if file_path.endswith(file_extension) and
                 file_path.startswith(start)]

    if len(files) > 1:
        print('Multiple files with the same start and end')
    file_path = directory / files[0]
    return file_path

def read_file(file_path):
    """
    Lazy reader of specific Neuralynx files
    :param file_path:
    :return: reader
    """
    reader = NeuralynxRawIO(filename=file_path)
    return reader


def get_event_times(folder, rescale=True):
    """
    Looks at just the events file for a Neuralynx data directory to get timestamps(default is seconds) and labels
    for recording events
    :param folder: string path
    :param rescale: optional(default True). Rescale timestamps to seconds from start of the file. Set to false for
    to get machine time.
    :return: event_times : I think this is in seconds from start if rescale=True,
    or in seconds in machine time if rescale=False.
    :return: event_labels : Whatever the annotation was for each event.
    :return: global_start : Machine code time beginning of recording(seconds) (only return if rescale=False)
    """
    # Obtained in seconds and assumes that the start of the file (not necessarily the task) is 0.
    all_files = os.listdir(folder)
    events_file = [file_path for file_path in all_files if file_path.startswith('Events')]
    if len(events_file) > 1:
        warnings.warn("More than one event file found.")
    elif len(events_file) == 0:
        warnings.warn("No events file found, Using Photodiode file to get global machine time start")
        event_times, event_labels = [], []
        ph_path = get_file_info(folder, "photo", ".ncs")
        ph_reader = read_file(ph_path)
        ph_reader.parse_header()
        global_start = ph_reader.global_t_start
    else:

        event_reader = read_file(os.path.join(folder, events_file[0]))
        event_reader.parse_header()
        global_start = event_reader.global_t_start
        try:
            event_timestamps, _, event_labels = event_reader.get_event_timestamps()
        except IndexError:
            warnings.warn("No events found")
            event_timestamps, event_labels = [], []
            ph_path = get_file_info(folder, "photo", ".ncs")
            ph_reader = read_file(ph_path)
            ph_reader.parse_header()
            global_start = ph_reader.global_t_start
        if rescale:
            if len(event_timestamps) > 0:
                event_times = event_reader.rescale_event_timestamp(event_timestamps)
            else:
                event_times = np.array([])
            global_start = None
        else:
            event_times = event_timestamps
    return event_times, event_labels, global_start


def missing_samples_check(file_path):
    """
    This script checks for missing samples in neuralynx files.
    :param file_path: Path object where data file is.
    :return: skipped_samples. integer number of skipped samples
    :return: t_starts. The start times for each segment
    :return: seg_sizes. The lengths of each Neo segment
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


def read_task_ncs(folder_name, file, task=None, events_file=None):
    """
    Read neuralynx data into an array, with sampling rate, and start time of the task.
    To deal with discontinuities and dropped samples, we take a pragmatic approach. We assume continuous sampling, and
    if there are inconsistencies between the number of samples in segments and the array itself, we fill in samples by
    interpolating.
    Ideally this spits out neuralynx data in the form of an array, with the sampling rate, and the start time of the task
    :param folder_name: Path object that tells the path of the structure
    :param file: filename we want to read currently
    :param task: (string, optional) that matches the event label in the actual events file. Ideally it matches the name of the task
    :param events_file: (string, optional) needed if task argument is provided, this used to be the .nev file but I
    found it useless so now it's a csv file that I generate via the scripts in manual_process
    :return: ncs_signal:
    :return: sampling_rate:
    :return: interp:
    :return: timestamps:
    """

    file_path = folder_name / file
    ncs_reader = read_file(file_path)
    ncs_reader.parse_header()
    n_segments = ncs_reader._nb_segment
    sampling_rate = ncs_reader.get_signal_sampling_rate()

    # This loop is to get around files that have weird events files, or task wasn't in the annotation
    if task is not None:
        # in previous iterations, we looked at the event file, in it's current form we'd rather look at the annotations
        # so first thing to check that the annotations file exists
        # we'll assume data is held in repo/data/task/subject/session/raw
        # but then that annotations file is held in parent direction

        labels_file = pd.read_csv(events_file)
        task_label = labels_file[labels_file.label == f"{task} duration"]
        task_start = task_label['time'].iloc[0].astype(float) # seconds from start of file
        task_end = task_start + task_label['duration'].iloc[0].astype(float)

        # print(float(ncs_reader.get_signal_size(block_index=0, seg_index=n_segments - 1))/sampling_rate)

        # The following block looks for the time of the start and end of the task we care about
        task_start_segment_index = None
        task_end_segment_index = None
        task_start_search = True
        for i in range(n_segments):
            time_segment_start = ncs_reader.get_signal_t_start(block_index=0, seg_index=i)
            # print(time_segment_start)
            if time_segment_start < task_start:
                continue
            elif (time_segment_start >= task_start) and (time_segment_start < task_end):
                # The first time this is run, the task_start_search bit flips
                if task_start_search:
                    task_start_search = False
                    # We take the index before because time_segment_start may not overlap with the start of the segment
                    # and this is looking from below, so overlap is with previous segment
                    task_start_segment_index = max(i - 1, 0)
            else:
                # The end isn't as important if we overshoot
                task_end_segment_index = i
                break
            if i == n_segments-1:
                task_end_segment_index = n_segments-1
    else:
        task_start_segment_index = 0
        task_end_segment_index = n_segments-1

    # I believe this is in number of seconds till start(if theoretically correct), the problem is that the sampling
    # rate is an average given to us by neuralynx
    task_start_segment_time = ncs_reader.get_signal_t_start(block_index=0, seg_index=task_start_segment_index)
    # Note the difference for getting task end
    task_end_segment_time = ncs_reader.segment_t_stop(block_index=0, seg_index=task_end_segment_index)

    array_size = (task_end_segment_time-task_start_segment_time) * sampling_rate
    timestamps = np.linspace(task_start_segment_time, task_end_segment_time, int(array_size))
    interp = np.zeros((int(array_size), ))
    ncs_signal = np.zeros((int(array_size), ))
    for i in range(task_start_segment_index, task_end_segment_index+1):
        # First stop. Get the time_segment_start and t_end for each segment.
        time_segment_start = ncs_reader.get_signal_t_start(block_index=0, seg_index=i)
        seg_size = ncs_reader.get_signal_size(block_index=0, seg_index=i)
        signal_segment = ncs_reader.get_analogsignal_chunk(seg_index=i)
        start_index = int((time_segment_start-ncs_reader.get_signal_t_start(block_index=0,
                                                                            seg_index=task_start_segment_index)) *
                          sampling_rate)
        # rescale to uV
        ncs_signal[start_index:start_index+seg_size] = ncs_reader.rescale_signal_raw_to_float(signal_segment,
                                                                                              dtype='float32').T[0]
        if i > 0:
            previous_segment_stop = ncs_reader.segment_t_stop(block_index=0, seg_index=i-1)
            if abs(time_segment_start-previous_segment_stop) < 1/sampling_rate:
                continue
            else:
                # 01/18/2024 - Consider a version of this script that doesn't interpolate, to allow for better alignment
                # to spike data where this has in fact not been done (spike sorting (Osort) doesn't care about
                # timestamping at all)
                previous_seg_signal = ncs_reader.get_analogsignal_chunk(seg_index=i-1)
                previous_seg_signal_scaled = ncs_reader.rescale_signal_raw_to_float(previous_seg_signal,
                                                                                    dtype='float32').T[0]
                previous_seg_time_start = ncs_reader.get_signal_t_start(block_index=0, seg_index=i-1)
                previous_seg_time_end = ncs_reader.segment_t_stop(block_index=0, seg_index=i-1)
                previous_seg_size_samples = ncs_reader.get_signal_size(block_index=0, seg_index=i-1)
                curr_seg_time_end = ncs_reader.segment_t_stop(block_index=0, seg_index=i)
                current_seg_signal_scaled = ncs_signal[start_index:start_index+seg_size]
                data_y = np.concatenate((previous_seg_signal_scaled, current_seg_signal_scaled))
                data_t = np.concatenate((np.linspace(previous_seg_time_start, previous_seg_time_end,
                                                     previous_seg_size_samples), np.linspace(time_segment_start,
                                                                                             curr_seg_time_end,
                                                                                             seg_size)))
                cs = CubicSpline(data_t, data_y)
                total_samples = int((curr_seg_time_end-previous_seg_time_start)*sampling_rate)
                full_data_t = np.linspace(previous_seg_time_start, curr_seg_time_end, total_samples)
                data_x_interp = cs(full_data_t)
                # Below pertains to the data_x_interp array made above
                missing_samples_start_ind = previous_seg_size_samples-1
                missing_samples_end_ind = int((time_segment_start-previous_seg_time_start)*sampling_rate)
                missing_samples = missing_samples_end_ind-missing_samples_start_ind

                ncs_signal[start_index-missing_samples:start_index] = data_x_interp[missing_samples_start_ind:
                                                                                    missing_samples_end_ind]
                interp[start_index-missing_samples:start_index] = np.ones((missing_samples_end_ind -
                                                                           missing_samples_start_ind, ))
    return ncs_signal, sampling_rate, interp, timestamps
