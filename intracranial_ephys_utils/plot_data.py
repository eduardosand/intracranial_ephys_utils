import matplotlib.pyplot as plt
import numpy as np


def diagnostic_time_series_plot(lfp_signal, freq, total_time=None, output_folder=None, electrode_name='',
                                subject=None, session=None):
    """
    Plot time series given a sampling rate to check for any weird idiosyncracies in data.
    Args:
        lfp_signal: (numpy array). The time series signal, ideal n_timepoints x 1.
        freq: (int). The sampling rate of the signal. To get a good estimate of time
        total_time: (float) (optional) Definite positive, used to limit the amount of time visualized.
        output_folder: (Path) (optional) Folder where to keep data.
        electrode_name: (string) (optional) Name of electrode if saving, or want a title
        subject: (string) (optional) Name of subject if saving
        session: (string) (optional) Name of session if saving

    Returns:

    """
    fig2, ax = plt.subplots(4, 1)

    ax[0].plot(np.linspace(0, 1, num=int(freq)), lfp_signal[0:int(freq)])
    ax[0].set_title(f'First second')
    midlevel_time = 30
    ax[1].plot(np.linspace(0, midlevel_time, num=int(freq*midlevel_time)), lfp_signal[0:int(midlevel_time*freq)])
    ax[1].set_title(f'First {midlevel_time} seconds')
    sec_in_minute = 60
    ax[2].plot(np.linspace(0, lfp_signal.shape[0]/freq/sec_in_minute, lfp_signal.shape[0]), lfp_signal)
    ax[2].set_title('Entire task in minutes')
    ax[3].plot(np.linspace(midlevel_time, 0, num=int(freq*midlevel_time)), lfp_signal[-int(midlevel_time*freq):])
    ax[3].set_title(f'Last {midlevel_time} seconds')

    for axis in ax:
        axis.set_ylabel('Voltage (uV)')
        axis.set_xlabel('Time (s)')
    ax[2].set_xlabel('Time (m)')
    if total_time is not None:
        ax[2].set_xlim([0, total_time])
    if electrode_name != '':
        plt.suptitle(f'Time Courses for {electrode_name}')
    if output_folder is not None:
        plt.savefig(output_folder / f'{subject}_{session}_{electrode_name}.png')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    return None