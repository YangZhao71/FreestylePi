import numpy as np
from os.path import isfile

from utils.functions import load_keras
from utils.audio import buffer_to_audio
from utils.vectorization import vectorize_raw, add_deltas
from utils.threshold_decoder import ThresholdDecoder


class TriggerDetector:
    """Class used to load the model and run trigger words detection
    """

    def __init__(self, model_path=None, chunk_size=2048,
                 sensitivity=0.5, trigger_level=3):
        if (model_path is None) or not isfile(model_path):
            raise FileNotFoundError("{} doesn't exist!".format(model_path))

        # Get rid of warnings!
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        from tensorflow.python.util import deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

        import tensorflow as tf

        self.model = self.load_model(model_path)
        self.graph = tf.get_default_graph()

        self.chunk_size = chunk_size
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level

        self.activation = 0

        self.pr = self.get_params()
        self.mfccs = np.zeros((self.pr['n_features'], self.pr['n_mfcc']))
        self.window_audio = np.array([])
        self.threshold_decoder = ThresholdDecoder(((6, 4),), 0.2)
        self.audio_buffer = np.zeros(self.pr['buffer_samples'], dtype=float)

    def get_params(self):
        """Get paramters
        """
        from math import floor
        window_t = 0.1
        hop_t = 0.05
        buffer_t = 1.5
        sample_rate = 16000
        hop_samples = int(sample_rate * hop_t + 0.5)
        window_samples = int(sample_rate * window_t + 0.5)
        buffer_samples = hop_samples * \
            ((int(sample_rate * buffer_t + 0.5)) // hop_samples)
        n_features = 1 + \
            int(floor((buffer_samples - window_samples) / hop_samples))
        return {'hop_samples':  hop_samples,
                'buffer_samples': buffer_samples,
                'window_samples': window_samples,
                'n_features': n_features, 'n_mfcc': 13}

    def load_model(self, model_path):
        """Load trained model with Keras
        """
        return load_keras().models.load_model(model_path)

    def predict(self, inputs):
        """Run model prediction
        """
        with self.graph.as_default():
            return self.model.predict(inputs)

    def run(self, inp):
        return self.predict(inp[np.newaxis])[0][0]

    def is_activate(self, prob):
        # type: (float) -> bool
        """Returns whether the new prediction caused an activation"""
        chunk_activated = prob > 1.0 - self.sensitivity

        if chunk_activated or self.activation < 0:
            self.activation += 1
            has_activated = self.activation > self.trigger_level
            if has_activated or chunk_activated and self.activation < 0:
                self.activation = -(8 * 2048) // self.chunk_size

            if has_activated:
                return True
        elif self.activation > 0:
            self.activation -= 1
        return False

    def update_vectors(self, stream):
        if isinstance(stream, np.ndarray):
            buffer_audio = stream
        else:
            if isinstance(stream, (bytes, bytearray)):
                chunk = stream
            else:
                chunk = stream.read(self.chunk_size)
            if len(chunk) == 0:
                raise EOFError
            buffer_audio = buffer_to_audio(chunk)

        self.window_audio = np.concatenate((self.window_audio, buffer_audio))

        if len(self.window_audio) >= self.pr['window_samples']:
            new_features = vectorize_raw(self.window_audio)
            self.window_audio = self.window_audio[len(
                new_features) * self.pr['hop_samples']:]
            if len(new_features) > len(self.mfccs):
                new_features = new_features[-len(self.mfccs):]
            self.mfccs = np.concatenate(
                (self.mfccs[len(new_features):], new_features))

        return self.mfccs

    def update(self, stream):
        mfccs = self.update_vectors(stream)
        raw_output = self.run(mfccs)
        return self.threshold_decoder.decode(raw_output)

    def get_prediction(self, chunk):
        audio = buffer_to_audio(chunk)
        self.audio_buffer = np.concatenate(
            (self.audio_buffer[len(audio):], audio))
        return self.update(chunk)


if __name__ == "__main__":
    pass
