from ephyviewer import mkQApp, MainViewer, TraceViewer, CsvEpochSource, EpochEncoder
import numpy as np
from pathlib import Path
from load_data import read_task_ncs, get_event_times
import os
import pandas as pd


def reformat_event_labels(subject, session, task, data_directory):
    """
    This script takes the events files, reads the timestamps in, and organizes them suitably for
    the data_viewer.
    :param subject:
    :param session:
    :param task:
    :param data_directory:
    :return:
    """
    data_dir = data_directory / subject / session
    event_times, event_labels, global_start = get_event_times(data_dir, rescale=False)
    event_times_sec, _, _ = get_event_times(data_dir, rescale=True)
    durations = np.ones((event_times_sec.shape[0], ))*0.5
    source_epoch = pd.DataFrame(np.array([event_times_sec, durations, event_labels]).T, columns=['time', 'duration', 'label'])
    if f'{subject}_{session}_{task}.csv' in os.listdir(data_dir):
        raise FileExistsError
    else:
        source_epoch.to_csv(data_dir / f'{subject}_{session}_{task}.csv', index=False)
    return None


def photodiode_check_viewer(subject, session, task, data_directory):
    """
    This script is a generalized dataviewer to look at a signal, and bring up the events
    :param subject:
    :param session:
    :param task:
    :param data_directory:
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
    filename = f'{subject}_{session}_{task}.csv'
    source_epoch = CsvEpochSource(os.path.join(data_dir, filename), possible_labels)

    # create a viewer for the encoder itself
    view2 = EpochEncoder(source=source_epoch, name='Tagging events')
    win.add_view(view2)

    #
    # view3 = EventList(source=source_epoch, name='events')
    # win.add_view(view3)

    # show main window and run Qapp
    win.show()

    app.exec()


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
