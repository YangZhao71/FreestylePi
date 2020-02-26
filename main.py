import sys
import time
import random
import queue
import threading

from termcolor import cprint
from utils.audio import ResumableMicrophoneStream
from utils.detect_queue import DetectQueue
from utils.credentials import init_credentials

from trigger_detector import TriggerDetector

from speech_to_text import SpeechToText
from lang_understand import LangUnderstand
from text_to_speech import TextToSpeech
from lyrics_generator import LyricsGenerator

from tft_display import TFTDisplay

# Audio recording parameters
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
STREAM_LIMIT = 5000


class Andrew(object):
    """the rap voice assisstant
    """
    def __init__(self, detect_model="data/andrew2.net",
                       lyrics_model="data/keras_model_1200.h5",
                       lyrics_chars="data/chars.pkl"):
        # microphone
        self.mic = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

        # wake word detector
        self.detector = TriggerDetector(detect_model)

        # speech and language services
        self.speech_client = SpeechToText()
        self.luis = LangUnderstand()
        self.tts = TextToSpeech()

        # lyrics generator model
        self.lyrics_gen = LyricsGenerator(lyrics_model, lyrics_chars)

        self.pred_queue = DetectQueue(maxlen=5)
        self.is_wakeup = False

        # pytft display
        self.tft = TFTDisplay()
        self.tft_queue = queue.Queue()
        self.tft_thread = threading.Thread(target=self.tft_manage, args=())
        self.tft_thread.daemon = True
        self.tft_thread.start()

        self.notify("hi_there")


    def notify(self, topic="hi_there", is_async=False, audio_path="data/audio"):
        # Notify with local preset audio files
        from os.path import join, isfile
        audio_file = join(audio_path, f"{topic}.wav")
        if not isfile(audio_file):
            return

        self.tts.play_file(audio_file, is_async)


    def generate_rap(self, topic="", beat_path="data/beat"):
        """Generate rap and play
        """
        tts = self.tts
        lyrics_gen = self.lyrics_gen

        response = tts.generate_speech(f"hey, I can rap about {topic}")
        tts.play(response, True)

        # Generate based on topic
        lyrics_output = lyrics_gen.generate(topic)

        # Generate speech
        lyrics_speech = tts.generate_speech(lyrics_output)

        # Select beat
        beat_index = random.randint(0, 20)

        # Play beat and lyrics
        tts.play_file(f'{beat_path}/beat_{beat_index}.wav', True)
        tts.play(lyrics_speech)

    def get_weather_message(self, city="Ithaca"):
        import requests, json, os
        api_key = os.getenv('WEATHER_APIKEY')
        base_url = "https://api.openweathermap.org/data/2.5/weather?"
        city_name = f"{city},us"
        complete_url = f"{base_url}q={city_name}&units=imperial&APPID={api_key}"
        try:
            response = requests.get(complete_url)
            res = response.json()
            msg_weather = f"Today, it's {res['weather'][0]['description']} in {city}. "
            msg_temp = f"The temperature is {int(res['main']['temp'])} degrees."
            return msg_weather + msg_temp
        except:
            pass

        return ""


    def intent_recognize(self, text=""):
        """Recognize intent
        """
        luis = self.luis
        tts = self.tts

        # Get result from language understanding engine
        luis_result = luis.predict(text)
        intent = luis_result.top_scoring_intent.intent

        if intent == "Freestyle":
            entities = luis_result.entities
            entity_topic = "rap"
            if (len(entities) > 0):
                entity = entities[0]
                cprint(f'The topic is {entity.entity}', 'cyan')
                entity_topic = entity.entity
            self.generate_rap(entity_topic)

        elif intent == "Weather":
            response = tts.generate_speech("I will tell you the weather in Ithaca.")
            tts.play(response)

            weather = self.get_weather_message()
            response = tts.generate_speech(weather)
            tts.play(response)

        else:
            self.notify("sorry")


    def tft_manage(self):
        """Manage TFT display through state
        """
        self.tft.display_text("Andrew is waking up")
        status = {'state': 'None'}

        while True:
            if status['state'] is 'wait':
                self.tft.display_wave()

            elif status['state'] is 'listen':
                self.tft.display_wave((0, 255, 0))

            # Update the status
            try:
                update = self.tft_queue.get(block=False)
                if update is not None:
                    status = update

            except queue.Empty:
                continue


    def start(self):
        """Start listening and interacting
        """
        tft = self.tft
        tts = self.tts

        # Init stream
        with self.mic as stream:

            self.tft_queue.put({'state': 'listen'})

            while True:
                if not self.is_wakeup:
                    stream.closed = False

                    while not stream.closed:

                        stream.audio_input = []
                        audio_gen = stream.generator()

                        for chunk in audio_gen:
                            if not self.is_wakeup:

                                prob = self.detector.get_prediction(chunk)

                                self.pred_queue.append(prob > 0.6)
                                print('!' if prob > 0.6 else '.', end='', flush=True)

                                if (self.pred_queue.count >= 2):
                                    self.notify("hi")
                                    cprint(' Trigger word detected! \n', 'magenta')
                                    self.pred_queue.clear()
                                    self.is_wakeup = True
                                    stream.pause()
                                    break
                else:
                    cprint('Speech to text\n', 'green')

                    time.sleep(1)
                    stream.closed = False

                    try:
                        voice_command = self.speech_client.recognize(stream)

                        cprint(f'{voice_command}\n', 'yellow')
                        cprint('Recognition ended...\n', 'red')

                        stream.pause()

                        #tft.display_text(f'"{voice_command}"')

                        if ("goodbye" in voice_command):
                            self.notify("see_you")
                            exit()

                        if ("sorry" in voice_command):
                            self.notify("its_ok")

                        else:
                            cprint('Recognize intents...', 'cyan')
                            self.intent_recognize(voice_command)

                    except Exception as e:
                        cprint(f'Error: {e}', 'red')

                    self.is_wakeup = False


def main():

    # set credentials for cloud services
    init_credentials()

    # init and start andrew
    andrew = Andrew()
    andrew.start()


if __name__ == "__main__":
    main()
