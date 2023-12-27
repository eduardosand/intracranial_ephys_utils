from ephyviewer import mkQApp, MainViewer, TraceViewer, CsvEpochSource, EpochEncoder
import numpy as np
from pathlib import Path
from .load_data import read_task_ncs, get_event_times
import os
import pandas as pd


def reformat_event_labels(subject, session, task, data_directory, result_directory):
    """
    This script takes the events files, reads the timestamps in, and organizes them suitably for
    the data_viewer.
    :param subject: (str). Subject
    :param session: (str). Session
    :param task: (srt). Task the subject completed in this session
    :param data_directory: (Path). The Path object that points to the neuralynx data directory
    :param result_directory: (Path). The Path object that points to where the annotations file will go.
    :return:
    """
    event_times, event_labels, global_start = get_event_times(data_directory, rescale=False)
    event_times_sec, _, _ = get_event_times(data_dir, rescale=True)
    durations = np.ones((event_times_sec.shape[0], ))*0.5
    source_epoch = pd.DataFrame(np.array([event_times_sec, durations, event_labels]).T, columns=['time', 'duration',
                                                                                                 'label'])
    if f'{subject}_{session}_{task}.csv' in os.listdir(result_directory):
        raise FileExistsError
    else:
        source_epoch.to_csv(result_directory / f'{subject}_{session}_{task}.csv', index=False)


def photodiode_check_viewer(subject, session, task, data_directory, results_directory):
    """
    This script is a generalized dataviewer to look at a signal, and bring up the events
    :param subject:
    :param session:
    :param task:
    :param data_directory:
    :param results_directory:
    :return:
    """

    data_dir = data_directory / subject / session
    ph_filename = 'photo1.ncs'
    ph_signal, sampling_rate, interp, timestamps = read_task_ncs(data_dir, ph_filename)
    dataset = np.expand_dims(ph_signal, axis=1)
    labels = np.expand_dims(np.array([ph_filename]), axis=1)
    t_start = 0.

    app = mkQApp()

    # Create main window that can contain several viewers
    win = MainViewer(debug=True, show_auto_scale=True)

    # Create a viewer for signal
    view1 = TraceViewer.from_numpy(dataset, sampling_rate, t_start, 'Photodiode', channel_names=labels)

    view1.params['scale_mode'] = 'same_for_all'
    view1.auto_scale()
    win.add_view(view1)

    possible_labels = [f'{task} duration']
    file_path = results_directory / f'{subject}_{session}_{task}.csv'
    source_epoch = CsvEpochSource(file_path, possible_labels)

    # create a viewer for the encoder itself
    view2 = EpochEncoder(source=source_epoch, name='Tagging events')
    win.add_view(view2)

    #
    # view3 = EventList(source=source_epoch, name='events')
    # win.add_view(view3)

    # show main window and run Qapp
    win.show()

    app.exec()


def write_timestamps(subject, session, task, annotations_directory, event_folder, local_data_directory):
    """
    Looks in event folders for labels. Elicits user input to determine which labels are relevant for spike sorting
    to constrain looking at only task-relevant data. User can input -1 if the whole datastream should be spike sorted.
    :param annotations_directory: This is the folder where manual annotations are found
    :param data_directory: This is the folder where events live (helpful to get global machine time)
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
    start_time_machine = start_time_sec*sec_microsec + global_start
    end_time_machine = end_time_sec*sec_microsec + global_start
    print(start_time_machine)
    timestamps_file = local_data_directory / f"timestampsInclude.txt"
    with open(timestamps_file, 'w') as f:
        f.write(f'{int(start_time_machine)}    {int(end_time_machine)}')


def su_timestamp_process(subject, session, task, data_directory, results_directory):
    """
    Master script that runs all timestamp writing in succession
    :param subject:
    :param session:
    :param task:
    :param data_directory:
    :param results_directory:
    :return:
    """
    reformat_event_labels(subject, session, task, data_directory, results_directory)

    photodiode_check_viewer(subject, session, task, data_directory, results_directory)

    data_dir = data_directory / subject / session
    write_timestamps(subject, session, task, results_directory, data_dir, results_directory)



def main():
    test_subject = 'IR87'
    test_session = 'sess-1'
    task = 'wcst'
    results_directory = Path("C:/Users/edsan/PycharmProjects/tt_su/data/")
    # task = 'Starting Recording'
    # task = 'WCST - begun'
    # task = 'wcst end'
    # reformat_event_labels(test_subject, test_session, task, results_directory)

    photodiode_check_viewer(test_subject, test_session, task, results_directory)

    # data_clean_viewer(test_subject, test_session, task, processed_data_directory, electrode_names, dataset, eff_fs[0])

    # plt.show()


if __name__ == "__main__":
    main()
