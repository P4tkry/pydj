import librosa
from effects import filter
import sounddevice as sd

# Load audio file with the appropriate sampling rate
data, fs = librosa.load('track2.mp3', sr=44100)

low = filter.low_filter(data, fs)
low = filter.add_echo(data, fs, 0.5, 0.5)
#
# high = filter.high_filter(data, fs)
#
# mid = filter.mid_filter(data, fs)
#
# low = low + high + mid
# normalize

# show plot



print('Odtwarzanie sygnału oryginalnego')
# Play the filtered signal
sd.play(low, fs)

sd.wait()