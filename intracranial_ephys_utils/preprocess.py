
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

def read_in_ncs_data_directory(file_paths, neuro_folder_name, task, high_pass=1000):
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


def save_data_for_Bob():
    return