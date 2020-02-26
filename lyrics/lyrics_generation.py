#!/usr/bin/python
#
# Author: Yangmengyuan Zhao (yz2453)
#
# Load the trained model and generate lyrics given user input
#
from keras.models import load_model
import sys
import numpy as np
import pickle

class LyricsModel(object):
    def __init__(self, model_path, chars_path):
        """Load the trained model"""
        self.model = load_model(model_path)
        with open(chars_path, 'rb') as f:
            self.chars = pickle.load(f)
    
    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_output(self, input='Hello'):
        generated = ''
        sentence = ('{0:0>40}').format(input).lower()
        generated += input 

        char_indices = dict((c, i) for i, c in enumerate(self.chars))
        indices_char = dict((i, c) for i, c in enumerate(self.chars))
        for i in range(400):

            x_pred = np.zeros((1, 40, len(self.chars)))

            for t, char in enumerate(sentence):
                if char != '0':
                    x_pred[0, t, char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, temperature = 1.0)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            if next_char == '\n':
                continue
        return generated

def main():
    model = LyricsModel('./keras_model_1200.h5', './chars.pkl')
    print('loaded model')
    output = model.generate_output()
    print(output)

if __name__ == "__main__":
    main()
