import os
import time
import sys

from google.cloud import speech_v1p1beta1 as speech
from six.moves import queue


class SpeechToText:
    """Wrapper for Google Speech API
    """

    def __init__(self, sample_rate=16000, stream_limit=10000):
        self.client = speech.SpeechClient()
        self.config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code='en-US',
            max_alternatives=1)
        self.streaming_config = speech.types.StreamingRecognitionConfig(
            config=self.config, interim_results=True, single_utterance=True)

        self.stream_limit = stream_limit
        self.requests = None
        self.responses = None

    def get_current_time(self):
        return int(round(time.time() * 1000))

    def recognize(self, stream, show_interim_results=True):
        self.requests = (speech.types.StreamingRecognizeRequest(
            audio_content=content) for content in stream.generator())

        self.responses = self.client.streaming_recognize(
            self.streaming_config, self.requests)

        num_chars_printed = 0
        for response in self.responses:
            if self.get_current_time() - stream.start_time > self.stream_limit:
                from termcolor import cprint
                cprint('Time exceeded', 'red')
                break

            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            if not result.is_final:
                if show_interim_results:
                    sys.stdout.write(transcript + overwrite_chars + '\r')
                    sys.stdout.flush()
                    num_chars_printed = len(transcript)
            else:
                return transcript

        return ""
