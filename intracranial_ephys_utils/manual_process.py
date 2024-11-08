from ephyviewer import mkQApp, MainViewer, TraceViewer, CsvEpochSource, EpochEncoder
import numpy as np
import warnings

from .load_data import read_task_ncs, get_event_times
from .plot_data import diagnostic_time_series_plot
from .preprocess import binarize_ph
import os
import pandas as pd


def reformat_event_labels(subject, session, task, data_directory, annotations_directory, extension=None):
    """
    This script takes the events files, reads the timestamps in, and organizes them suitably for
    the data_viewer. Outputs the events as a csv file
    :param subject: (str). Subject
    :param session: (str). Session
    :param task: (srt). Task the subject completed in this session
    :param data_directory: (Path). The Path object that points to the neuralynx data directory
    :param annotations_directory: (Path). The Path object that points to where the annotations file will go.
    :param extension: str optional.
    :return:
    """
    event_times, event_labels, _, event_file = get_event_times(data_directory, rescale=False, extension=extension)
    print(event_file)
    event_times_sec, _, _, _ = get_event_times(data_directory, rescale=True, extension=extension)
    if len(event_times) == 0:
        source_epoch = pd.DataFrame(np.array([[], [], []]).T, columns=['time', 'duration', 'label'])
    else:
        durations = np.ones((event_times_sec.shape[0], ))*0.5
        source_epoch = pd.DataFrame(np.array([event_times_sec, durations, event_labels]).T, columns=['time', 'duration',
                                                                                                     'label'])

    file_root, _ = os.path.splitext(event_file)
    annotations_file = f'{subject}_{session}_{task}_{file_root}.csv'
    if annotations_file in os.listdir(annotations_directory):
        print('Annotations File exists, so nothing was written. Double check to see if it matches expectation.')
    else:
        source_epoch.to_csv(annotations_directory / annotations_file, index=False)
    return event_file


def photodiode_check_viewer(subject, session, task, data_directory, annotations_directory, diagnostic=False,
                            task_start=0., events_filename=None):
    """
    This script is a generalized dataviewer to look at a photodiode signal, and bring up the events.
    With the viewer, we can make annotations and save them to a csv file. Additionally, if the diagnostic optional
    parameter is set to True, this function will also preprocess the photodiode to check that the signal on TTL is good.
    :param subject: (string) The patient ID
    :param session: (string) Session of the experiment. Useful if patient completed more than one session of a task.
    :param task: (string) Which task
    :param data_directory: (Path object) Where the data lives
    :param annotations_directory: (Path object) Where we want to put the annotations of the data
    :param diagnostic: (bool) (optional) If True we will also plot diagnostics, and preprocess photodiode and overlay
    it.
    :param task_start: (float) (optional) The start time of the task
    :param events_filename: (str) (optional) The extension of the file (in case multiple datasets in the same folder)
    :return:
    """

    # Underlying assumption is that data is organized as subject/session/raw, this is where the .ncs files live
    all_files_list = os.listdir(data_directory)
    # electrode_files = [file_path for file_path in all_files_list if (re.match('m.*ncs', file_path) and not
    #                file_path.endswith(".nse"))]

    ext = events_filename.replace('.nev', '.ncs')
    ext = ext.replace('events', '')
    ext = ext.replace('Events', '')
    if ext is not None and ext != '.ncs':
        warnings.warn(f"Most likely multiple photodiode files, picking {ext} now")
        ph_files = [file_path for file_path in all_files_list if file_path.endswith(ext) and
                    (file_path.startswith('photo1') or file_path.startswith('Photo'))]
    else:
        ph_files = [file_path for file_path in all_files_list if file_path.endswith('.ncs') and
                    (file_path.startswith('photo1') or file_path.startswith('Photo'))]
        if len(ph_files) > 1:
            warnings.warn("Multiple photodiode files picking the base one now")
            ph_files = [file for file in ph_files if (file.endswith('photo1.ncs') or file.endswith('photo.ncs'))]
    assert len(ph_files) == 1
    ph_filename = ph_files[0]

    # We'll read in the photodiode signal
    ph_signal, sampling_rate, interp, timestamps = read_task_ncs(data_directory, ph_filename)

    # we'll make some sanity plots to check photodiode at a glance, and preprocess to double-check event trigger
    # is good
    if diagnostic:
        diagnostic_time_series_plot(ph_signal, sampling_rate, electrode_name='Photodiode')
        # Use the diagnostic plot to limit the time interval we process photodiode in
        start_time = input('Type start time (in seconds) if not the start of signal, else press enter: ')
        end_time = input('Type end time (in seconds) if not the end of signal, else press enter: ')
        if len(start_time) == 0:
            start_time = 0
        else:
            start_time = int(start_time)
        if len(end_time) == 0:
            end_time = int(ph_signal.shape[0])
        else:
            end_time = int(end_time)

        ph_signal = ph_signal[int(start_time*sampling_rate):int(sampling_rate * end_time)]
        # timestamps = timestamps[int(start_time*sampling_rate):int(sampling_rate * end_time)]
        t_start = start_time

        # next step to this is to add my thresholding for photodiode
        ph_signal_bin = binarize_ph(ph_signal, sampling_rate)
        dataset = np.vstack([ph_signal, ph_signal_bin]).T
        labels = np.array([ph_filename, 'Photodiode Binarized'])
    else:
        t_start = task_start
        dataset = np.expand_dims(ph_signal, axis=1)
        labels = np.expand_dims(np.array([ph_filename]), axis=1)

    app = mkQApp()

    # Create main window that can contain several viewers
    win = MainViewer(debug=True, show_auto_scale=True)

    # Create a viewer for signal
    # T_start essentially rereferences the start time of the dataset, but leaves the annotations alone
    # be wary of this
    view1 = TraceViewer.from_numpy(dataset, sampling_rate, t_start, 'Photodiode', channel_names=labels)

    # TO DO
    # Figure out a better way to scale the different signals when presented
    # view1.params['scale_mode'] = 'same_for_all'
    view1.auto_scale()
    win.add_view(view1)

    # annotation file details
    possible_labels = [f'{task} duration']
    file_root, _ = os.path.splitext(events_filename)
    file_path = annotations_directory / f'{subject}_{session}_{task}_{file_root}.csv'
    source_epoch = CsvEpochSource(file_path, possible_labels)

    # create a viewer for the encoder itself
    view2 = EpochEncoder(source=source_epoch, name='Tagging events')
    win.add_view(view2)

    # show main window and run Qapp
    win.show()
    app.exec()


def data_clean_viewer(subject, session, task, annotations_directory, electrode_names, dataset, fs):
    """
    This function serves to look at the microwire signals and look at which is the reference or to look at the
    macrowires and clean the data for epileptic activity
    Current: Only microwire functionality so far
    Assumes dataset
    :param subject: (string) subject id
    :param session: (string) session id
    :param task: (string) task id
    :param annotations_directory:
    :param electrode_names: (1d array)
    :param dataset: (n_electrodes, n_timepoints)
    :param fs: (int)
    :return:
    """
    possible_labels = ['bad epileptic activity macro', 'bad eight hertz noise', 'bad epileptic activity micro',
                       'bad epileptic activity both', 'reference electrode', 'microPED electrode',
                       'white noise electrode', 'clipping noise electrode', 'epileptic macrocontact',
                       'white noise macrocontact']
    file_path = annotations_directory / f'{subject}_{session}_{task}_post_timestamping_events.csv'
    source_epoch = CsvEpochSource(file_path, possible_labels)

    t_start = 0.
    # you must first create a main Qt application (for event loop)
    app = mkQApp()

    # Create the main window that can contain several viewers
    win = MainViewer(debug=True, show_auto_scale=True)

    # create a viewer for signal
    view1 = TraceViewer.from_numpy(dataset.T, fs, t_start, 'Microwires', channel_names=electrode_names)
    # view1 = TraceViewer.from_neo_analogsignal(analog_signals, 'sigs')
    view1.params['scale_mode'] = 'same_for_all'
    view1.auto_scale()
    win.add_view(view1)

    # create a viewer for the encoder itself
    view2 = EpochEncoder(source=source_epoch, name='Tagging events')
    win.add_view(view2)

    #
    # view3 = EventList(source=source_epoch, name='events')
    # win.add_view(view3)
    # show main window and run Qapp
    win.show()

    app.exec()


def write_timestamps(subject, session, task, event_folder, annotations_directory, local_data_directory,
                     events_filename=None):
    """
    Looks in event folders for labels. Elicits user input to determine which labels are relevant for spike sorting
    to constrain looking at only task-relevant data. User can input -1 if the whole datastream should be spike sorted.
    :param subject: (string) Subject identifier
    :param session: (string) Session identifier (1 if subject only completed one run of the experiment.
    :param task: (string) Task identifier. The task or experiment subject completed.
    :param annotations_directory: This is the folder where manual annotations are found
    :param event_folder: This is the folder where events live (helpful to get global machine time)
    :param local_data_directory:  This is where the microwire data to be sorted is
    :param events_filename: optional. If multiple events
    :return: None. A txt file is generated with relative timestamps if needed, or not if not needed.
    """
    if events_filename is None:
        file_root = 'events'
        labels_file = pd.read_csv(annotations_directory / f'{subject}_{session}_{task}_events.csv')
    else:
        file_root, _ = os.path.splitext(events_filename)
        labels_file = pd.read_csv(annotations_directory / f'{subject}_{session}_{task}_{file_root}.csv')
    task_label = labels_file[labels_file.label == f"{task} duration"]
    print(task_label['time'].iloc[0])
    start_time_sec = task_label['time'].iloc[0].astype(float)
    end_time_sec = start_time_sec + task_label['duration'].iloc[0].astype(float)

    # this exists in case there are multiple events file, and we want to pick the right one
    if events_filename is None:
        ext = None
    else:
        ext = events_filename[-6:]
    _, _, global_start, _ = get_event_times(event_folder, rescale=False, extension=ext)
    # microsec_sec = 10**-6
    sec_microsec = 10**6
    start_time_machine = (start_time_sec + global_start) * sec_microsec
    end_time_machine = (end_time_sec + global_start) * sec_microsec
    timestamps_file = local_data_directory / f"timestampsInclude.txt"
    specified_file = local_data_directory / f"which_files.txt"
    # generate directory if it doesn't exist
    if not os.path.exists(local_data_directory):
        os.mkdir(local_data_directory)
    with open(timestamps_file, 'w+') as f:
        f.write(f'{int(start_time_machine)}    {int(end_time_machine)}')
    with open(specified_file, 'w+') as f:
        f.write(f'{file_root}')


def su_timestamp_process(subject, session, task, data_directory, annotations_directory, results_directory):
    """
    Master script that runs basic pipeline to get timestamps file for sorting minimally using OSort Matlab scripts.
    :param subject: (str). Subject identifier
    :param session: (str). Session identifier, useful if subject ran more than one session.
    :param task: (str). Task identifier. The task the subject ran.
    :param data_directory: (Path). Path object to data where events file and photodiode file lives
    :param annotations_directory: (Path). Path object that points to where we'd like to store annotations and metadata.
    :param results_directory: (Path). Path object that points to where the timestampInclude.txt file will end up.
    """
    # double check if there are multiple events files
    _, _, global_start, event_files = get_event_times(data_directory, rescale=False)
    if len(event_files) > 1 and type(event_files) == type([]):
        print('Multiple Events Files, we will go through the separate datasets one at a time')
        print(event_files)
        for event_file in event_files:
            print(event_file)
            ext = event_file[-6:]
            print(ext)
            reformat_event_labels(subject, session, task, data_directory, annotations_directory, extension=ext)
            photodiode_check_viewer(subject, session, task, data_directory, annotations_directory, diagnostic=False,
                                    events_filename=event_file)
            file_root, _ = os.path.splitext(event_file)
            labels_file = pd.read_csv(annotations_directory / f'{subject}_{session}_{task}_{file_root}.csv')
            task_label = labels_file[labels_file.label == f"{task} duration"]
            if len(task_label) == 0:
                continue
            else:
                target_file = event_file
                break
    else:
        reformat_event_labels(subject, session, task, data_directory, annotations_directory)
        photodiode_check_viewer(subject, session, task, data_directory, annotations_directory, diagnostic=False,
                                events_filename=event_files)
        file_root, _ = os.path.splitext(event_files)
        labels_file = pd.read_csv(annotations_directory / f'{subject}_{session}_{task}_{file_root}.csv')
        task_label = labels_file[labels_file.label == f"{task} duration"]
        target_file = event_files
    write_timestamps(subject, session, task, data_directory, annotations_directory, results_directory,
                     events_filename=target_file)


def get_annotated_task_start_time(subject, session, task, annotations_directory):
    """
    This function serves as a helper function to grab the start and end times of the task, after annotating the
    photodiode script. Will only work if annotations file exists, and duration event has been made.
    Recall that loading in .ncs files relies on loading in many segments of variable length, so we typically load in
    more data than what start and end time would have you believe
    :param subject:
    :param session:
    :param task:
    :param annotations_directory:
    :return: start_time_sec (float) Start time in seconds for the task. Reference is the start of the file recording.
    :return: end_time_sec (float) End time in seconds for the task. Reference is the start of the file recording.
    :return: duration (float) Duration in seconds for the task.
    """
    labels_file = pd.read_csv(annotations_directory / f'{subject}_{session}_{task}_events.csv')
    task_label = labels_file[labels_file.label == f"{task} duration"]
    start_time_sec = task_label['time'].iloc[0].astype(float)
    end_time_sec = start_time_sec + task_label['duration'].iloc[0].astype(float)
    return start_time_sec, end_time_sec, task_label['duration'].iloc[0].astype(float)
