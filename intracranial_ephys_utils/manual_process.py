from ephyviewer import mkQApp, MainViewer, TraceViewer, CsvEpochSource, EpochEncoder
import numpy as np
from pathlib import Path
from .load_data import read_task_ncs, get_event_times
import os
import pandas as pd


def reformat_event_labels(subject, session, task, data_directory, annotations_directory):
    """
    This script takes the events files, reads the timestamps in, and organizes them suitably for
    the data_viewer.
    :param subject: (str). Subject
    :param session: (str). Session
    :param task: (srt). Task the subject completed in this session
    :param data_directory: (Path). The Path object that points to the neuralynx data directory
    :param annotations_directory: (Path). The Path object that points to where the annotations file will go.
    :return:
    """
    event_times, event_labels, global_start = get_event_times(data_directory, rescale=False)
    event_times_sec, _, _ = get_event_times(data_directory, rescale=True)
    durations = np.ones((event_times_sec.shape[0], ))*0.5
    source_epoch = pd.DataFrame(np.array([event_times_sec, durations, event_labels]).T, columns=['time', 'duration',
                                                                                                 'label'])
    annotations_file = f'{subject}_{session}_{task}_events.csv'
    if annotations_file in os.listdir(annotations_directory):
        print('Annotations File exists, double check')
    else:
        source_epoch.to_csv(annotations_directory / annotations_file, index=False)


def photodiode_check_viewer(subject, session, task, data_directory, annotations_directory, task_start=0.):
    """
    This script is a generalized dataviewer to look at a photodiode signal, and bring up the events
    :param subject: The patient ID
    :param session: Session of the experiment. Useful if patient completed more than one session of a task.
    :param task: Which task
    :param data_directory: Where the data lives
    :param annotations_directory: Where we want to put the annotations of the data
    :return:
    """

    all_files_list = os.listdir(data_directory)
    # electrode_files = [file_path for file_path in all_files_list if (re.match('m.*ncs', file_path) and not
    #                file_path.endswith(".nse"))]
    ph_files = [file_path for file_path in all_files_list if file_path.endswith('.ncs') and
                       file_path.startswith('photo1')]
    assert len(ph_files) == 1
    ph_filename = ph_files[0]
    print(ph_filename)
    ph_signal, sampling_rate, interp, timestamps = read_task_ncs(data_directory, ph_filename)
    # end_signal = 1470
    # ph_signal = ph_signal[:int(sampling_rate * end_signal)]
    # timestamps = timestamps[:int(sampling_rate * end_signal)]
    dataset = np.expand_dims(ph_signal, axis=1)
    labels = np.expand_dims(np.array([ph_filename]), axis=1)
    # On occasion, there are long recordings, which make it difficult to see what's going on, especially if there are
    # large discontinuities...
    t_start = task_start

    app = mkQApp()

    # Create main window that can contain several viewers
    win = MainViewer(debug=True, show_auto_scale=True)

    # Create a viewer for signal
    view1 = TraceViewer.from_numpy(dataset, sampling_rate, t_start, 'Photodiode', channel_names=labels)

    view1.params['scale_mode'] = 'same_for_all'
    view1.auto_scale()
    win.add_view(view1)

    possible_labels = [f'{task} duration']
    file_path = annotations_directory / f'{subject}_{session}_{task}_events.csv'
    source_epoch = CsvEpochSource(file_path, possible_labels)
    print('okay')
    # create a viewer for the encoder itself
    view2 = EpochEncoder(source=source_epoch, name='Tagging events')
    win.add_view(view2)
    print('huh')
    #
    # view3 = EventList(source=source_epoch, name='events')
    # win.add_view(view3)

    # show main window and run Qapp
    win.show()

    app.exec()


def write_timestamps(subject, session, task, event_folder, annotations_directory, local_data_directory):
    """
    Looks in event folders for labels. Elicits user input to determine which labels are relevant for spike sorting
    to constrain looking at only task-relevant data. User can input -1 if the whole datastream should be spike sorted.
    :param subject: (string) Subject identifier
    :param session: (string) Session identifier (1 if subject only completed one run of the experiment.
    :param task: (string) Task identifier. The task or experiment subject completed.
    :param annotations_directory: This is the folder where manual annotations are found
    :param event_folder: This is the folder where events live (helpful to get global machine time)
    :param local_data_directory:  This is where the microwire data to be sorted is
    :return: None. A txt file is generated with relative timestamps if needed, or not if not needed.
    """
    labels_file = pd.read_csv(annotations_directory / f'{subject}_{session}_{task}.csv')
    task_label = labels_file[labels_file.label == f"{task} duration"]
    print(task_label['time'].iloc[0])
    start_time_sec = task_label['time'].iloc[0].astype(float)
    end_time_sec = start_time_sec + task_label['duration'].iloc[0].astype(float)
    _, _, global_start = get_event_times(event_folder, rescale=False)
    microsec_sec = 10**-6
    sec_microsec = 10**6
    start_time_machine = (start_time_sec + global_start) * sec_microsec
    end_time_machine = (end_time_sec + global_start) * sec_microsec
    timestamps_file = local_data_directory / f"timestampsInclude.txt"
    with open(timestamps_file, 'w') as f:
        f.write(f'{int(start_time_machine)}    {int(end_time_machine)}')


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
    reformat_event_labels(subject, session, task, data_directory, annotations_directory)
    photodiode_check_viewer(subject, session, task, data_directory, annotations_directory)
    write_timestamps(subject, session, task, data_directory, annotations_directory, results_directory)
