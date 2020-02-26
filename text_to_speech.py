import os
from google.cloud import texttospeech


class TextToSpeech:
    """Text-to-Speech wrapper
    """

    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

        self.voice = texttospeech.types.VoiceSelectionParams(
            name='en-US-Wavenet-B',
            language_code='en-US',
            ssml_gender=texttospeech.enums.SsmlVoiceGender.MALE)

        self.audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            pitch=4.0,
            speaking_rate=1.2)

    def generate_speech(self, text):
        synthesis_input = texttospeech.types.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            synthesis_input, self.voice, self.audio_config)
        return response.audio_content

    @staticmethod
    def play(audio=None, is_async=False):
        """Play raw bytes audio
        """
        from utils.audio import play_bytes_alsa
        if is_async:
            import threading
            t = threading.Thread(name='play', target=play_bytes_alsa, args=(audio,))
            t.start()
        else:
            play_bytes_alsa(audio)

    @staticmethod
    def play_file(filename=None, is_async=False):
        from utils.audio import play_bytes_alsa
        audio = open(filename, 'rb').read()

        if is_async:
            import threading
            t = threading.Thread(name='play', target=play_bytes_alsa, args=(audio,))
            t.start()
        else:
            play_bytes_alsa(audio)


if __name__ == "__main__":
    from utils.credentials import init_credentials
    init_credentials()

    from utils.audio import save_audio

    tts = TextToSpeech()
    generated = tts.generate_speech("Hi!")
    tts.play(generated)
    file_name = "data/audio/hi.wav"

    with open(file_name, "wb") as out:
        out.write(generated)

    #generated = tts.generate_speech("Yo! His palms are sweaty, knees weak, arms are heavy. There's vomit on his sweater already, mom's spaghetti")
    #tts.play_file('data/output.wav')
    #tts.play(generated)
