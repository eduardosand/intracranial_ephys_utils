from .load_data import read_task_ncs
from scipy import signal
import numpy as np
import os
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import warnings
import mne


def otsu_intraclass_variance(image, threshold):
    """
    Otsu's intra-class variance.
    If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.
    """
    return np.nansum(
        [
            np.mean(cls) * np.var(image, where=cls)
            #   weight   ·  intra-class variance
            for cls in [image >= threshold, image < threshold]
        ]
    )
    # NaNs only arise if the class is empty, in which case the contribution should be zero, which `nansum` accomplishes.


def otsu_threshold(time_series):
    """
    Otsu thresholding. I know it's for an image, but it should get the job done here in this time series signal, since
    it quite literally is two classes with noise. What difference does it make it if the variation in foreground and
    background happen in time than space.
    :param time_series:
    :return:
    """
    otsu_threshold = min(np.linspace(np.min(time_series), np.max(time_series),100),
        key=lambda th: otsu_intraclass_variance(time_series, th),
    )
    return otsu_threshold


def binarize_ph(ph_signal, sampling_rate, cutoff_fraction=2, task_time=None, tau=None):
    '''
    Binarizes the photodiode signal using the midpoint of the signal. Use local midpoints if given a tau.
    New version of this script uses a high pass filter and then peaks to find timepoints at which the signal changes by
    a lot and fast. We then try to find the exact timepoints when this happens by localizing runs where the high pass
    filter result is away from 0. We find the change in average signal before and after these timepoints to get a
    a sense of whether it's an event onset or offset. Finally, we assume we'll find some noise so we threshold these
    onsets and offsets by the point of greatest difference in changes (heuristic for Otsu threshold)
    :param ph_signal: This is the photodiode signal itself (array of floats)
    :param sampling_rate: How many samples per second (float)
    :param cutoff_fraction: What is the cutoff for on times. 2 is the midpoint (midpoint through the process of
    turning off or on)
    :param task_time: This is how long the task took (in minutes). Helpful to zoom in on particular data regions (float)
    :return: ph_signal_bin: Array of event_lengths in seconds, to be plotted with the rts (array of floats)
    '''
    if task_time is not None:
        total_time = int(task_time*sampling_rate*60)
    else:
        total_time = ph_signal.shape[0]
    ph_signal_bin = np.zeros((total_time, ))
    # 15 hz for ir95
    sos = signal.butter(4, 20, 'hp', fs=sampling_rate, output='sos')
    filtered = signal.sosfiltfilt(sos, ph_signal)
    stdev = np.std(filtered)
    # The idea with this method of processing, slightly more complicated than midpoint
    # is to take points significantly far away from 4 * stdeviations of the high passed signal
    # Then sequentially take pre and post averages to dictate signs. If sign is positive, then the value after this
    # is 1, otherwise 0.

    # ph_signal = filtered
    timepoints = np.arange(ph_signal.shape[0])
    if tau is not None:
        tau_samples = tau*sampling_rate
        partitions = (total_time // tau_samples) + 1
        print('huh')
        # Okay so calculating a rolling midpoint for even a 10 minute segment will take forever with
        # a sampling rate of several thoughsand
        midpoint = np.array([[(max(ph_signal[i*int(tau_samples):int(tau_samples)*(i+1)])-
                     min(ph_signal[i*int(tau_samples):int(tau_samples)*(i+1)]))/cutoff_fraction+
                    min(ph_signal[i*int(tau_samples):int(tau_samples)*(i+1)])]*int(tau_samples)
                    for i in range(int(partitions))]).flatten()
        midpoint = midpoint[0:total_time]
        ph_signal_bin[ph_signal[0:total_time] > midpoint] = 1.
    else:
        # issue with this function in low signal regime
        # with IR95 session 3, had to use 4 std dev
        # for IR94 session 1, signal is feeble, need to use a different thing
        events = timepoints[abs(filtered) > 2.15 * stdev]
        print(len(events))
        buffer = 0.02*sampling_rate
        sample_size = int(0.045*sampling_rate)
        event_breakpoint = 0
        event_onsets_initial = []
        event_offsets_initial = []

        sign_changes = []
        for i, sample_num in enumerate(events):
            # deal with multiple events and looping
            if sample_num <= event_breakpoint:
                continue
            nearby_occurrences = events[(events < sample_num + buffer) & (events > sample_num - buffer)]
            num_events = len(nearby_occurrences)
            if num_events > 1:
                avg_sample_num = int(np.median(nearby_occurrences))
                event_breakpoint = np.max(nearby_occurrences)
                max_sample_num = int(np.max(nearby_occurrences))
                min_sample_num = int(np.min(nearby_occurrences))
                # if max_sample_num - min_sample_num > sample_size:
                sign_change = (np.average(ph_signal[max_sample_num:min(len(ph_signal)-1,max_sample_num+sample_size)]) -
                           np.average(ph_signal[max(0, min_sample_num-sample_size):min_sample_num]))
                print('not okay')
            else:
                avg_sample_num = nearby_occurrences[0]
                sign_change = (np.average(ph_signal[avg_sample_num:min(len(ph_signal)-1, avg_sample_num+sample_size)]) -
                               np.average(ph_signal[max(0, avg_sample_num-sample_size):avg_sample_num]))
                print('okay')
            # print(sign_change)
            # plt.plot(ph_signal[avg_sample_num-sample_size:avg_sample_num+sample_size])
            # plt.show()
            if sign_change > 0:
                # positive, event_onset
                event_onsets_initial.append([avg_sample_num, abs(sign_change)])
            else:
                event_offsets_initial.append([avg_sample_num, abs(sign_change)])
            sign_changes.append(abs(sign_change))

        print(len(sign_changes))
        plt.hist(sign_changes)
        plt.title('Histogram of sign changes')
        print('why')
        # drop_ind only works in cases where there is a clear separation, however it the signal is smeared, it no longer
        # works
        # drop_ind = np.argmax(np.diff(np.sort(sign_changes)))
        sign_change_drop = otsu_threshold(sign_changes)

        # sign_change_drop = np.sort(sign_changes)[drop_ind+1]
        event_onsets = np.array(event_onsets_initial)
        event_offsets = np.array(event_offsets_initial)
        event_onsets = event_onsets[event_onsets[:,1] > sign_change_drop, 0]
        event_offsets = event_offsets[event_offsets[:,1] > sign_change_drop, 0]
        plt.show()
        print('weird')
        # Now we have all onsets and offsets, recreate our binarized signals using this
        # first check that these are the same length, and that the first event_onset is first
        if (len(event_onsets) == len(event_offsets)) and (event_onsets[0] < event_offsets[0]):
            print('Events detected are the same size and make sense')
        elif len(event_onsets) == len(event_offsets):
            print('On events and off events do not have the same size')
        else:
            print('Issue with start or stop, check manual timestamping')

        for i, event_onset in enumerate(event_onsets):
            possible_offsets = event_offsets[event_offsets>event_onset]
            best_offset = np.min(possible_offsets)
            ph_signal_bin[int(event_onset): int(best_offset)] = 1.
        # midpoint = ((max(ph_signal[0:total_time])-min(ph_signal[0:total_time]))/cutoff_fraction+
        #             min(ph_signal[0:total_time]))
        # ph_signal_bin[ph_signal[0:total_time] > midpoint] = 1.
    return ph_signal_bin


def BCI_LFP_processing(lfp_signals, sampling_rate):
    """
    This code is meant to resample BCI data down to a numpy array of a more managable sampling, and get
    rid of power line noise
    :param lfp_signals:
    :param sampling_rate:
    :return:
    """
    # Will take SU 30K Hz down to 1K
    # Will take iEEG macrocontacts from 2K down to 1K
    if sampling_rate == 30000:
        first_factor = 3
        low_freq = 30
        high_freq = 3000
        # bandpass for spikes down to 10K, and then decimate after
        downsampled_signal = signal.decimate(lfp_signals, first_factor, axis=1)
        effective_fs = sampling_rate/first_factor
        butterworth_bandpass = signal.butter(4, (low_freq, high_freq), 'bp', fs=effective_fs, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, downsampled_signal, axis=1)
        second_factor = 10
        downsampled_signal_2 = signal.decimate(bandpass_signal, second_factor, axis=1)
        effective_fs /= second_factor
    elif sampling_rate == 2000:
        first_factor = 2
        low_freq = 0.1
        high_freq = 900
        # bandpass for spikes down to 1K, and then decimate after
        butterworth_bandpass = signal.butter(4, (low_freq, high_freq), 'bp', fs=sampling_rate, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, lfp_signals, axis=1)
        downsampled_signal_2 = signal.decimate(bandpass_signal, first_factor, axis=1)
        effective_fs = sampling_rate/first_factor
    # print(effective_fs)
    f0 = 60.
    Q = 30.0  # Quality Factor
    b_notch, a_notch = signal.iirnotch(f0, Q, effective_fs)
    processed_signals = signal.filtfilt(b_notch, a_notch, downsampled_signal_2, axis=1)
    # Get harmonics out of the signal as well, up to 300
    for i in range(2, 6):
        b_notch, a_notch = signal.iirnotch(f0*i, Q, effective_fs)
        processed_signals = signal.filtfilt(b_notch, a_notch, processed_signals, axis=1)
    return processed_signals, effective_fs


def broadband_seeg_processing(lfp_signals, sampling_rate, lowfreq, high_freq):
    """
    This function takes in an lfp signal and performs basic processing. This function is janky but that's only
    because decimations of too big a factor result in artifacts, so I need one solution for 32K, and a different one
    for other sampling rates.
    Preprocessing is as follows, downsample once. We want to make sure we stay above Nyquist limit, so then we run our
    a 4th order Butterworth to bandpass from lowfreq to highfreq. Downsample once more to get down to 1KHz sampling rate
    Finally, we'll pass through a Notch filter to get rid of powerline noise and associated harmonics.
    WARNING: This is NOT good for European data because of the power line noise there(its 50Hz)
    :param lfp_signals: (np.array) (1, n_samples)
    :param sampling_rate: (int)
    :param lowfreq: (int) lower frequency for bandpass
    :param high_freq: (int) higher frequency for bandpass
    :return: processed_signals: numpy array shape (1, n_samples)
    :return: effective_fs: (int) Final sampling rate after processing
    """
    if sampling_rate == 32000:
        if high_freq == 1000:
            second_factor = 8
        elif high_freq == 2000:
            second_factor = 4
        else:
            second_factor = 8
        first_factor = 4
        # for microLFP, this brings us down to 8KHz
        downsampled_signal = signal.decimate(lfp_signals, first_factor)
        effective_fs = sampling_rate/first_factor
        butterworth_bandpass = signal.butter(4, (lowfreq, high_freq), 'bp', fs=effective_fs, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, downsampled_signal)
        # for microLFP, this brings us down to 1 Khz
        downsampled_signal_2 = signal.decimate(bandpass_signal, second_factor)
        effective_fs /= second_factor
    elif sampling_rate == 8000:
        if high_freq == 1000:
            second_factor = 8
        elif high_freq == 2000:
            second_factor = 4
        else:
            second_factor = 8
        # first_factor = 1
        butterworth_bandpass = signal.butter(4, (lowfreq, high_freq), 'bp', fs=sampling_rate, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, lfp_signals)
        # for macroLFP, this brings us down to 1 Khz
        downsampled_signal_2 = signal.decimate(bandpass_signal, second_factor)
        effective_fs = sampling_rate/second_factor
    elif sampling_rate == 4000:
        if high_freq == 1000:
            second_factor = 4
        elif high_freq == 2000:
            second_factor = 2
        else:
            second_factor = 4
        # first_factor = 1
        butterworth_bandpass = signal.butter(4, (lowfreq, high_freq), 'bp', fs=sampling_rate, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, lfp_signals)
        # for macroLFP, this brings us down to 1 Khz
        downsampled_signal_2 = signal.decimate(bandpass_signal, second_factor)
        effective_fs = sampling_rate / second_factor
    else:
        raise Exception('Invalid sampling rate')

    f0 = 60.
    q = 30.0  # Quality Factor
    b_notch, a_notch = signal.iirnotch(f0, q, effective_fs)
    processed_signals = signal.filtfilt(b_notch, a_notch, downsampled_signal_2)

    # Get harmonics out of the signal as well, up to 300
    for i in range(2, 6):
        b_notch, a_notch = signal.iirnotch(f0*i, q, effective_fs)
        processed_signals = signal.filtfilt(b_notch, a_notch, processed_signals)
    return processed_signals, int(effective_fs)


def preprocess_dataset(file_paths, neuro_folder_name, low_pass=1000, task=None, events_file=None):
    """
    Read in all data from a given directory and run basic preprocessing on it so I can load it live on my shitty
    computer.
    :param file_paths: (list) A list of filenames. Ex(['LAC1.ncs','LAC2.ncs'])
    :param neuro_folder_name: (Path) The folderpath where the data is held
    :param task: (optional) A string that dictates the task name, only use if you have the event labels already, so already parsed
    through the photodiode file and annotated the task duration
    :param events_file: (optional) (Path) Where the annotation file is located, needed if task is given.
    :param low_pass: the largest frequency to use for band-pass filtering
    :return: dataset: Numpy array, shape is (n_channels, n_samples)
    :return: eff_fs: Effective sampling rate
    :return: electrode_names: List of electrode names
    """
    eff_fs = []
    electrode_names = []
    for ind, micro_file_path in enumerate(file_paths):
        print(micro_file_path)
        split_tup = os.path.splitext(micro_file_path)
        ncs_filename = split_tup[0]
        if ncs_filename.startswith('photo'):
            lfp_signal, sample_rate, interp, timestamps = read_task_ncs(neuro_folder_name, micro_file_path, task=task,
                                                                        events_file=events_file)
            print('timestamps below')
            print(timestamps)
            # assume photo is 8K and we're getting down to 1000
            if low_pass == 1000:
                first_factor = 8
            elif low_pass == 2000:
                first_factor = 4
            else:
                first_factor = 8
            fs = sample_rate / first_factor
            processed_lfp = signal.decimate(lfp_signal, first_factor)
            downsampled_timestamps = timestamps[::first_factor]
            # processed_timestamps = signal.decimate(timestamps, first_factor)
            print('sliced timestamps below')
            print(downsampled_timestamps)
        else:
            lfp_signal, sample_rate, _, _ = read_task_ncs(neuro_folder_name, micro_file_path, task=task,
                                                          events_file=events_file)
            processed_lfp, fs = broadband_seeg_processing(lfp_signal, sample_rate, 0.1, low_pass)
        if ind == 0:
            dataset = np.zeros((len(file_paths)+1, processed_lfp.shape[0]))
            dataset[ind, :] = processed_lfp
            eff_fs.append(fs)
            electrode_names.append(ncs_filename)
            og_file = micro_file_path
        else:
            # Currently the loading of photodiode is 3 ms different in size(it's more than the others)
            if processed_lfp.shape[0] > dataset.shape[1]:
                print(f'{micro_file_path} array is larger than {og_file}')
                print(processed_lfp.shape)
                print(dataset.shape)
                dataset[ind, 0:dataset.shape[1]] = processed_lfp[0:dataset.shape[1]]
            else:
                dataset[ind, :processed_lfp.shape[0]] = processed_lfp[0:processed_lfp.shape[0]]
            eff_fs.append(fs)
            electrode_names.append(ncs_filename)
        if ncs_filename.startswith('photo'):
            dataset[-1, :] = downsampled_timestamps
    eff_fs.append(fs)
    electrode_names.append('Timepoints')
    return dataset, eff_fs, electrode_names


def save_small_dataset(subject, session, task_name, events_file, low_pass=1000):
    """
    Load, process, and savedata.
    :param subject: (string) subject identifier
    :param session: (string) subject session
    :param task_name: (string) task name, used to select only part of entire ncs file, assuming annotations file exists
    :param events_file: (Path) path to where events annotation file is located
    :param low_pass: (int) specify
    :return:
    """
    # Hopefully your file structure is like mine
    data_directory = Path(f"{os.pardir}/data/{subject}/{session}/raw")
    results_directory = data_directory.parent.absolute() / "preprocessed"
    print(results_directory)
    if results_directory.exists():
        print('Results Directory already Exists')
    else:
        os.mkdir(results_directory)
    all_files_list = os.listdir(data_directory)
    # electrode_files = [file_path for file_path in all_files_list if (re.match('m.*ncs', file_path) and not
    #                file_path.endswith(".nse"))]
    electrode_files = [file_path for file_path in all_files_list if file_path.endswith('.ncs')]
    electrode_files.sort()
    # electrode_files.append('photo1.ncs')
    dataset, eff_fs, electrode_names = preprocess_dataset(electrode_files, data_directory, task=task_name,
                                                          events_file=events_file, low_pass=low_pass)
    if len(set(eff_fs)) != 1:
        warnings.warn('Different effective sampling rates across files')
    bp = str(int(eff_fs[0]))
    np.savez(os.path.join(results_directory, f'{subject}_{session}_{task_name}_lowpass_{bp}'), dataset=dataset,
             electrode_names=electrode_names, eff_fs=eff_fs)
    return None


def save_as_npy(subject, session, task_name, data_directory, events_file, electrode_selection, one_file=False):
    """
    Load data from neuralynx files and package them into .npy files. No preprocessing done to the data, so microwires,
    and macrocontacts at different sampling rates, treated separately.
    :param subject: (string) subject identifier
    :param session: (string) session identifier
    :param task_name: (string) task identifier
    :param data_directory: (Path) path object that tells us where the raw data lives (if in the cluster, it won't be in
    our expected data/subject/session style, hence why this function is the way it is)
    :param events_file: (Path) path object that tells us where the events file, ideally the events_file contains one
    event titled f"{task_name} duration"
    :param electrode_selection: (string) Whether to save macrocontact or microwire data
    :param one_file: (optional) (bool) whether to package data into one file. If false, package data into different files
    :return:
    """

    # Hopefully your file structure is like mine
    results_directory = Path(f"{os.pardir}/data/{subject}/{session}/preprocessed")
    print(results_directory)
    if results_directory.exists():
        print('Results Directory already Exists')
    else:
        os.mkdir(results_directory)
    all_files_list = os.listdir(data_directory)
    if electrode_selection == "microwire":
        print(all_files_list)
        electrode_files = [file_path for file_path in all_files_list if file_path.endswith('.ncs')
                           and file_path.startswith('m') and not file_path.startswith('mic1')]
        electrode_files.sort()
        eff_fs = []
        electrode_names = []
        print(electrode_files)
        for ind, micro_file_path in enumerate(electrode_files):
            print(micro_file_path)
            split_tup = os.path.splitext(micro_file_path)
            ncs_filename = split_tup[0]
            lfp_signal, sample_rate, interp, timestamps = read_task_ncs(data_directory, micro_file_path,
                                                                        task=task_name,
                                                                        events_file=events_file)
            if one_file:
                if ind == 0:
                    dataset = np.zeros((len(electrode_files) + 1, lfp_signal.shape[0]))
                    dataset[ind, :] = lfp_signal
                    eff_fs.append(sample_rate)
                    electrode_names.append(ncs_filename)
                else:
                    dataset[ind, :] = lfp_signal
                    eff_fs.append(sample_rate)
                    electrode_names.append(ncs_filename)
            else:
                dataset = lfp_signal
                bp = str(int(sample_rate))
                np.savez(os.path.join(results_directory, f'{subject}_{session}_{task_name}_{ncs_filename}_{bp}'),
                         dataset=dataset, electrode_name=ncs_filename, fs=sample_rate, timestamps=timestamps)
    elif electrode_selection == "macrocontact":
        raise NotImplementedError
        ########### TO DO
        # the function will be the same, but just don't know how to do the electrode selection (maybe use lazy reader
        # to exclude files with a certain sample rate?
    if one_file:
        bp = str(int(eff_fs[0]))
        np.savez(os.path.join(results_directory, f'{subject}_{session}_{task_name}_{electrode_selection}_{bp}'),
                 dataset=dataset, electrode_names=electrode_names, eff_fs=eff_fs, timestamps=timestamps)
    return None


def make_trialwise_data(event_times, electrode_names, fs, dataset, tmin=-1., tmax=1., baseline=None, annotations=None):
    """
    This function serves to convert a dataset that is from start to stop, into one that is organized by trials.
    :param event_times: (timestamps for trial onsets, offsets, or anything of interest)
    :param electrode_names: (list) list of strings that contain the name for each signal
    :param fs: (int) sampling rate
    :param dataset: (np.array) raw data
    :param tmin: (opt)
    :param tmax: (opt)
    :param baseline: (opt) Tuple that defines the period to use as baseline
    :param annotations: mne annotations object
    :return: epochs_object
    """

    # The safest way to do this is to build a mne object, first step is to create the info for that object
    events = np.zeros((event_times.shape[0], 3))
    events[:, 0] = event_times
    events[:, 2] = np.ones((event_times.shape[0],))[:] #rule_codes commented out for now because i need something to run
    events = events.astype(int)
    mne_info = mne.create_info(electrode_names, fs, ch_types='seeg')
    raw_data = mne.io.RawArray(dataset, mne_info)
    if annotations is not None:
        raw_data.set_annotations(annotations)

    num_electrodes, num_samples = dataset.shape
    # Trying to compare some things
    epochs_object = mne.Epochs(raw_data, events, tmax=tmax, tmin=tmin, baseline=baseline,
                               reject_by_annotation=False)
    trial_based_data = epochs_object.get_data(copy=True)
    epochs_object = mne.Epochs(raw_data, events, tmax=tmax, tmin=tmin, baseline=baseline,
                               reject_by_annotation=True)
    return epochs_object, trial_based_data


def smooth_data(data, fs, window, step):
    """
    Smooth data by taking the average in windows, and stepping by some amount of time
    Expects data to be 3D (number of trials X number of electrodes X number of timepoints)
    We will smooth by taking the centered average about a window, so the smoothed data will be smaller than expected
    :param data: np.array
    :param fs: (int) sampling rate
    :param window: (float) in seconds, how much to average over, the larger this is the more our signal is smeared.
    :param step: (float) in seconds. How much to step forward, determines new_fs
    :return: smoothed_data, new_fs
    """
    # first create array that is the processed shape
    num_epochs, num_electrodes, num_timepoints = data.shape
    # why should our smoothed data be this?
    # we'd like to smooth data by taking averages with a window size and moving by a certain step
    # ideally window is centered, effectively meaning that we can only take as many timepoints that equal to
    # (num_timepoints/fs - step) / step and this simplifies to below
    smoothed_data = np.zeros((num_epochs, num_electrodes, int(num_timepoints/(fs*step)-1)))
    for i in range(smoothed_data.shape[2]):
        # print(i)
        if i == 0:
            smoothed_data[:, :, i] = np.mean(data[:, :, i:int((i+1)*window*fs)], axis=2)
        elif (i*step*fs + window*fs) > num_timepoints:
            print('There is an issue with this code and the estimation of the smoothed data size')
            start = i*step*fs - window/2
            smoothed_data[:, :, i] = np.mean(data[:, :, int(start): num_timepoints], axis=2)
        else:
            # print('int')
            # print(data.shape)
            # print(i * step * fs)
            start = i * step * fs
            smoothed_data[:, :, i] = np.mean(data[:, :, int(start-window/2*fs): int(start + window/2*fs)], axis=2)
    new_fs = 1 / step
    return smoothed_data, new_fs