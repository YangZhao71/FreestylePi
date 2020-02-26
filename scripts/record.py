"""Script to record short audio wavfiles with pyaudio.

NOTE: This module requires the dependencies `pyaudio`.
To install using pip:

    pip install pyaudio

Example usage:
    python record.py [-d duration]
"""

import argparse
import sys
from time import time
from os.path import isfile

import wave
from pyaudio import PyAudio


def record_until(p, should_return, args):
    chunk_size = 1024
    stream = p.open(format=p.get_format_from_width(args.width), channels=args.channels,
                    rate=args.rate, input=True, frames_per_buffer=chunk_size)

    frames = []
    while not should_return():
        frames.append(stream.read(chunk_size))

    stream.stop_stream()
    stream.close()

    return b''.join(frames)


def save_audio(name, data, args):
    wf = wave.open(name, 'wb')
    wf.setnchannels(args.channels)
    wf.setsampwidth(args.width)
    wf.setframerate(args.rate)
    wf.writeframes(data)
    wf.close()


def next_name(name):
    def get_name(i):
        return "{}-{}.wav".format(name, str(i).zfill(2))
    i = 0
    while True:
        if not isfile(get_name(i)):
            break
        i += 1
    return get_name(i)


def main():
    parser = argparse.ArgumentParser(
        'Record audio training and testing samples')
    parser.add_argument('--width', '-w', default=2,
                        help='Sample width of audio')
    parser.add_argument('--rate', '-r', default=16000,
                        help='Sample rate of audio')
    parser.add_argument('--channels', '-c', default=1,
                        help='Number of audio channels')
    parser.add_argument('--duration', '-d', default=3,
                        help='Duration of the recorded audio')

    args = parser.parse_args()

    args.file_label = input("File label: ")
    pa = PyAudio()

    while (True):
        input("Press enter to record next {}-sec piece...".format(args.duration))

        begin = time()
        count = 0

        def should_return():
            delta = time() - begin
            sys.stdout.write("{0:.2f}\r".format(delta))
            sys.stdout.flush()
            return delta > args.duration

        name = next_name(args.file_label)
        d = record_until(pa, should_return, args)
        save_audio(name, d, args)
        print('Saved as ' + name)


if __name__ == '__main__':
    main()
