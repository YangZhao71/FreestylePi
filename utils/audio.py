"""Audio-related classes and functions
"""
import time
import numpy as np
import pyaudio
from six.moves import queue
from typing import *

# Audio recording parameters
STREAMING_LIMIT = 10000
# SAMPLE_RATE = 16000
# CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=16000, chunk_size=1600):
        self._rate = rate
        self._chunk_size = chunk_size
        self._num_channels = 1
        self._max_replay_secs = 5

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = self.get_current_time()
        self.audio_input = []

        # 2 bytes in 16 bit samples
        self._bytes_per_sample = 2 * self._num_channels
        self._bytes_per_second = self._rate * self._bytes_per_sample

        self._bytes_per_chunk = (self._chunk_size * self._bytes_per_sample)
        self._chunks_per_second = (self._bytes_per_second // self._bytes_per_chunk)

    def __enter__(self):
        self.closed = False

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def get_current_time(self):
        return int(round(time.time() * 1000))

    def pause(self):
        self.start_time = self.get_current_time()
        self.closed = True
        self._buff.queue.clear()

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def play_bytes_pyaudio(audio):
    # Play audio bytes with pyaudio
    import wave, io, pyaudio
    wf = wave.open(io.BytesIO(audio), 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()


def play_bytes_alsa(audio):
    # Play audio bytes with alsaaudio
    import wave, io, alsaaudio

    f = wave.open(io.BytesIO(audio), 'rb')
    device = alsaaudio.PCM(device='plughw:CARD=ALSA,DEV=0')
    device.setchannels(f.getnchannels())
    device.setrate(f.getframerate())
    device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    periodsize = f.getframerate() // 8

    device.setperiodsize(periodsize)

    data = f.readframes(periodsize)

    while data:
        device.write(data)
        data = f.readframes(periodsize)


class InvalidAudio(ValueError):
    pass


def chunk_audio(audio: np.ndarray, chunk_size: int) -> Generator[np.ndarray, None, None]:
    for i in range(chunk_size, len(audio), chunk_size):
        yield audio[i - chunk_size:i]


def buffer_to_audio(buffer: bytes) -> np.ndarray:
    """Convert a raw mono audio byte string to numpy array of floats"""
    return np.fromstring(buffer, dtype='<i2').astype(np.float32, order='C') / 32768.0


def load_audio(file, sample_rate=16000):
    """
    Args:
        file: Audio filename or file object
    Returns:
        samples: Sample rate and audio samples from 0..1
    """
    import wavio
    import wave
    try:
        wav = wavio.read(file)
    except (EOFError, wave.Error):
        wav = wavio.Wav(np.array([[]], dtype=np.int16), 16000, 2)
    if wav.data.dtype != np.int16:
        raise InvalidAudio('Unsupported data type: ' + str(wav.data.dtype))
    if wav.rate != sample_rate:
        raise InvalidAudio('Unsupported sample rate: ' + str(wav.rate))

    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)


def save_audio(filename, audio, sample_rate=16000, sample_depth=2):
    import wavio
    save_audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    wavio.write(filename, save_audio, sample_rate,
                sampwidth=sample_depth, scale='none')


