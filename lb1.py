import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter


class SourceAudio:
    played: np.ndarray = np.array([])

    def __init__(self, name, samplerate=None):
        self.audio, self.sr = librosa.load(name, sr=samplerate)
        self.audio = librosa.util.normalize(self.audio)
        self.bpm = int(librosa.beat.beat_track(y=self.audio, sr=self.sr)[0])

    def __bool__(self):
        return len(self.audio) > 0

    @property
    def current_time(self):
        return len(self.played) / self.sr

    def get_segment(self, count):
        if len(self.audio) < count:
            segment = self.audio
            self.audio = np.array([])
        else:
            segment = self.audio[:count]
            self.audio = self.audio[count:]

        self.played = np.append(self.played, segment)

        return segment


class Audio:
    bpm: int
    low: float = 1
    mid: float = 1
    high: float = 1

    def __init__(self, name, samplerate=None):
        self.source = SourceAudio(name, samplerate)
        self.bpm = self.source.bpm

    def __bool__(self):
        return bool(self.source)

    @property
    def current_time(self):
        return self.source.current_time

    def change_bpm(self, audio):
        # Calculate the time-stretch factor based on the initial and target BPM
        time_stretch_factor = self.source.bpm / self.bpm

        # Calculate the new length of the audio signal
        new_length = int(len(audio) * time_stretch_factor)

        # Resample the audio signal to the new length using interpolation
        resampled_audio = np.interp(
            np.linspace(0, len(audio), new_length, dtype=np.float32),
            np.arange(len(audio), dtype=np.float32),
            audio.astype(np.float32)
        )
        return resampled_audio.astype(np.float32)


    def adjust_volume(self, segment):
        nyquist = self.source.sr / 2

        low_normal_cutoff = 250 / nyquist
        b_low, a_low = butter(N=2, Wn=low_normal_cutoff, btype='low', analog=False)

        # mid_normal_cutoff = [250 / nyquist, 4000 / nyquist]
        # b_mid, a_mid = butter(N=2, Wn=mid_normal_cutoff, btype='band', analog=False)
        #
        # high_normal_cutoff = 4000 / nyquist
        # b_high, a_high = butter(N=2, Wn=high_normal_cutoff, btype='high', analog=False)
        #
        # low_signal = np.convolve(segment, b_low, mode='same').astype(np.float32)
        # mid_signal = np.convolve(segment, b_mid, mode='same').astype(np.float32)
        # high_signal = np.convolve(segment, b_high, mode='same').astype(np.float32)

        # adjusted_signal = (low_signal * self.low + mid_signal * self.mid + high_signal * self.high)

        adjusted_signal = np.convolve(segment, b_low, mode='same').astype(np.float32)
        adjusted_signal *= self.low

        return adjusted_signal


    def get_segment(self, count=1000):
        segment = self.source.get_segment(count)
        segment = self.change_bpm(segment)
        # segment = self.adjust_volume(segment)

        return segment


