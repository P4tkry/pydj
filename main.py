import librosa
import sounddevice as sd
import threading
import soundfile as sf
from lb1 import Audio
import numpy as np

audio = Audio('track4.mp3')


# show audio plot
#

def play_audio():
    global audio
    with sd.Stream(samplerate=audio.source.sr, channels=1, dtype='float32') as stream:
        while audio:
            segment = audio.get_segment()

            # print(audio.current_time)
            stream.write(segment)



# Load audio files


# Create threads for audio playback
thread1 = threading.Thread(target=play_audio, args=())

# Start the threads
thread1.start()
print('Audio playback started')
print('Current BPM:', audio.bpm)
while True:
    low = float(input('Enter new bpm: '))
    audio.low = low
    print('BPM changed to', low)

# Wait for both threads to finish
thread1.join()