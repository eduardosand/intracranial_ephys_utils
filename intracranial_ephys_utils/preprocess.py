from .load_data import read_task_ncs
from scipy import signal
import numpy as np
import os
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt


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
    sos = signal.butter(4, 15, 'hp', fs=sampling_rate, output='sos')
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
        events = timepoints[abs(filtered) > 4. * stdev]
        buffer = 0.02*sampling_rate
        sample_size = int(0.045*sampling_rate)
        event_breakpoint = 0
        event_onsets = []
        event_offsets = []

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
            else:
                avg_sample_num = nearby_occurrences[0]
            max_sample_num = int(np.max(nearby_occurrences))
            min_sample_num = int(np.min(nearby_occurrences))
            sign_change = (np.average(ph_signal[max_sample_num:max_sample_num+sample_size]) -
                           np.average(ph_signal[min_sample_num-sample_size:min_sample_num]))
            # print(sign_change)
            # plt.plot(ph_signal[avg_sample_num-sample_size:avg_sample_num+sample_size])
            # plt.show()
            if sign_change > 0:
                # positive, event_onset
                event_onsets.append([avg_sample_num, abs(sign_change)])
            else:
                event_offsets.append([avg_sample_num, abs(sign_change)])
            sign_changes.append(abs(sign_change))


        plt.hist(sign_changes)
        plt.title('Histogram of sign changes')
        drop_ind = np.argmax(np.diff(np.sort(sign_changes)))
        sign_change_drop = np.sort(sign_changes)[drop_ind+1]
        event_onsets = np.array(event_onsets)
        event_offsets = np.array(event_offsets)
        event_onsets = event_onsets[event_onsets[:,1] > sign_change_drop, 0]
        event_offsets = event_offsets[event_offsets[:,1] > sign_change_drop, 0]
        plt.show()
        # Now we have all onsets and offsets, recreate our binarized signals using this
        # first check that these are the same length, and that the first event_onset is first
        if (len(event_onsets) == len(event_offsets)) and (event_onsets[0] < event_offsets[0]):
            print('Events detected are the same size and make sense')
        else:
            print('welllllll shit')
            # print(event_onsets[0] < event_offsets[0])
            # print(event_onsets[-1] < event_offsets[-1])
            # print(len(event_onsets))
            # print(len(event_offsets))
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


def broadband_seeg_processing(lfp_signals, sampling_rate, lowfreq, highfreq):
    """
    This function takes in an lfp signal and performs basic processing. This function is janky but that's only
    because decimations of too big a factor result in artifacts, so I need one solution for 32K, and a different one
    for other sampling rates.
    Preprocessing is as follows, downsample once. We want to make sure we stay above Nyquit limit, so then we run our
    a 4th order Butterworth to bandpass from lowfreq to highfreq. Downsample once more to get down to 1KHz sampling rate
    Finally, we'll pass through a Notch filter to get rid of powerline noise and associated harmonics.
    WARNING: This is NOT good for European data because of the power line noise there(its 50Hz)
    :param lfp_signals:
    :param sampling_rate:
    :param lowfreq: lower frequency for bandpass
    :param highfreq: higher frequency for bandpass
    :return: processed_signals: numpy array shape (1, n_samples)
    :return: effective_fs: Final sampling rate after processing
    """
    if sampling_rate == 32000:
        first_factor = 4
        # for microLFP, this brings us down to 8KHz
        downsampled_signal = signal.decimate(lfp_signals, first_factor)
        effective_fs = sampling_rate/first_factor
        butterworth_bandpass = signal.butter(4, (lowfreq, highfreq), 'bp', fs=effective_fs, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, downsampled_signal)
        second_factor = 8
        # for microLFP, this brings us down to 1 Khz
        downsampled_signal_2 = signal.decimate(bandpass_signal, second_factor)
        effective_fs /= second_factor
    elif sampling_rate == 8000:
        # first_factor = 1
        butterworth_bandpass = signal.butter(4, (lowfreq, highfreq), 'bp', fs=sampling_rate, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, lfp_signals)
        second_factor = 8
        # for macroLFP, this brings us down to 1 Khz
        downsampled_signal_2 = signal.decimate(bandpass_signal, second_factor)
        effective_fs = sampling_rate/second_factor
    elif sampling_rate == 4000:
        # what if sampling rate is 4K
        # first_factor = 1
        butterworth_bandpass = signal.butter(4, (lowfreq, highfreq), 'bp', fs=sampling_rate, output='sos')
        bandpass_signal = signal.sosfiltfilt(butterworth_bandpass, lfp_signals)
        second_factor = 4
        # for macroLFP, this brings us down to 1 Khz
        downsampled_signal_2 = signal.decimate(bandpass_signal, second_factor)
        effective_fs = sampling_rate / second_factor
    else:
        raise Exception('Invalid sampling rate')
    f0 = 60.
    Q = 30.0  # Quality Factor
    b_notch, a_notch = signal.iirnotch(f0, Q, effective_fs)
    processed_signals = signal.filtfilt(b_notch, a_notch, downsampled_signal_2)
    # Get harmonics out of the signal as well, up to 300
    for i in range(2, 6):
        b_notch, a_notch = signal.iirnotch(f0*i, Q, effective_fs)
        processed_signals = signal.filtfilt(b_notch, a_notch, processed_signals)
    return processed_signals, effective_fs

def read_in_ncs_data_directory(file_paths, neuro_folder_name, task=None,high_pass=1000):
    """
    Read in all data from a given directory and run basic preprocessing on it so I can load it live on my shitty
    computer.
    :param file_paths: A list of filenames. Ex(['LAC1.ncs','LAC2.ncs'])
    :param neuro_folder_name: The folderpath where the data is held
    :param task: A string that dictates the task name, only use if you have the event labels already, so already parsed
    through the photodiode file and annotated the task duration
    :param high_pass: the largest frequency to use for band-pass filtering
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
        lfp_signal, sample_rate, _ = read_task_ncs(neuro_folder_name, micro_file_path, task=task)
        if ncs_filename.startswith('photo'):
            # assume photo is 8K and we're getting down to 1000
            first_factor = 8
            fs = sample_rate / first_factor
            processed_lfp = signal.decimate(lfp_signal, first_factor)
        elif (ncs_filename.startswith('m') and not ncs_filename.startswith('mic')):
            # processed_lfp, fs = broadband_seeg_processing(lfp_signal, sample_rate, 0.1, 1000)
            processed_lfp, fs = broadband_seeg_processing(lfp_signal, sample_rate, 0.1, high_pass)
        else:
            # processed_lfp, fs = broadband_seeg_processing(lfp_signal, sample_rate, 0.1, 1000)
            processed_lfp, fs = broadband_seeg_processing(lfp_signal, sample_rate, 0.1, high_pass)
        if ind == 0:
            dataset = np.zeros((len(file_paths), processed_lfp.shape[0]))
            dataset[ind, :] = processed_lfp
            eff_fs.append(fs)
            electrode_names.append(ncs_filename)
        else:
            # Currently the loading of photodiode is 3 ms different in size(it's more than the others)
            print(processed_lfp.shape[0])
            print(dataset.shape)
            if processed_lfp.shape[0] > dataset.shape[1]:
                dataset[ind, 0:dataset.shape[1]] = processed_lfp[0:dataset.shape[1]]
            else:
                print(processed_lfp.shape)
                print(dataset.shape)
                dataset[ind, :processed_lfp.shape[0]] = processed_lfp[0:processed_lfp.shape[0]]
            eff_fs.append(fs)
            electrode_names.append(ncs_filename)
    return dataset, eff_fs, electrode_names


def save_data_for_Bob_clean(subject, session, task_name):
    """
    Load, process, and savedata.
    :param subject:
    :param session:
    :param task_name:
    :return:
    """
    # Hopefully your file structure is like mine
    data_directory = Path(f"{os.pardir}/data/{subject}/{session}/raw")
    results_directory = data_directory.parent.absolute() / "Bob_viewer"
    if results_directory.exists():
        print('great')
    else:
        os.mkdir(results_directory)
    all_files_list = os.listdir(data_directory)
    # electrode_files = [file_path for file_path in all_files_list if (re.match('m.*ncs', file_path) and not
    #                file_path.endswith(".nse"))]
    electrode_files = [file_path for file_path in all_files_list if file_path.endswith('.ncs')]
    electrode_files.sort()
    # electrode_files.append('photo1.ncs')
    dataset, eff_fs, electrode_names = read_in_ncs_data_directory(electrode_files, data_directory)
    bp = str(int(eff_fs))
    np.savez(os.path.join(results_directory, f'{subject}_{session}_{task_name}_{bp}_broadband'), dataset=dataset,
             electrode_names=electrode_names, eff_fs=eff_fs)
    return None
