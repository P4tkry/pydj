import librosa
import numpy as np
from scipy import signal


def low_filter(sample, fs=44100, boost_value=1):
    b_low, a_low = signal.butter(4, 300, 'low', fs=fs)
    sample = signal.filtfilt(b_low, a_low, sample)
    return boost(sample, boost_value)


def mid_filter(sample, fs=44100, boost_value=1):
    b_mid, a_mid = signal.butter(4, [300, 2000], 'band', fs=fs)
    sample = signal.filtfilt(b_mid, a_mid, sample)
    return boost(sample, boost_value)


def high_filter(sample, fs=44100, boost_value=1):
    b_high, a_high = signal.butter(4, 2000, 'high', fs=fs)
    sample = signal.filtfilt(b_high, a_high, sample)
    return boost(sample, boost_value)


def boost(sample, boost=1):
    sample = sample * boost
    return sample


def add_echo(audio_data, sample_rate, delay, decay):
    """
    Add echo to an audio signal using librosa.

    Parameters:
    - audio_data: The original audio signal (numpy array).
    - sample_rate: The sample rate of the audio.
    - delay: The delay in seconds between the original sound and the echo.
    - decay: The factor by which the echo is scaled.

    Returns:
    - The audio signal with echo.
    """
    # Convert delay from seconds to samples
    delay_samples = int(sample_rate * delay)

    # Create a new array to hold the echoed signal
    echoed_signal = np.copy(audio_data)

    # Add the echo
    for i in range(delay_samples, len(audio_data)):
        echoed_signal[i] += decay * audio_data[i - delay_samples]

    # Clip values to prevent overflow in the audio signal
    echoed_signal = np.clip(echoed_signal, -1.0, 1.0)

    return echoed_signal
