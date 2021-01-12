"""Machine learning model for programming identification"""

import os
from os import path
import gc
import logging
from pathlib import Path
from math import ceil
import json
import joblib

import numpy as np
import tensorflow as tf

from FeatureExtract import extract, CONTENT_SIZE

from Proccess import (search_files, extract_from_files, read_file)

# Settings list
# LOGGER = logging.getLogger(__name__)

_NEURAL_NETWORK_HIDDEN_LAYERS = [256, 64, 16]
_OPTIMIZER_STEP = 0.05

_FITTING_FACTOR = 20
_CHUNK_PROPORTION = 0.2
_CHUNK_SIZE = 1000

class Predictor:

    def __init__(self, model_dir=None):

        self.checkpoint_path = model_dir
        
        #: supported languages with associated extensions
        with open('config/languages.json') as f:
            self.languages = json.load(f)

        n_classes = len(self.languages)

        self._checkpoint_path = model_dir

        self._classifier = tf.keras.models.Sequential()
        self._classifier.add(tf.keras.layers.Dense(256, input_dim=1024, activation='relu'))
        self._classifier.add(tf.keras.layers.Dense(64, activation='relu'))
        self._classifier.add(tf.keras.layers.Dense(16, activation='relu'))
        self._classifier.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        self._classifier.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.checkpoint_path,
            verbose = 1,
            save_weights_only = True,
            save_freq = 5
        )
       
    def language(self, text):
        # predict language name
        values = extract(text)
        self._classifier.load_weights(self.checkpoint_path)
        proba = self._classifier.predict([values])
        proba = proba.tolist()

        # Order the languages from the most probable to the least probable
        positions = np.argsort(proba)[0][::-1]
        names = np.sort(list(self.languages))
        names = names[positions]
        
        return names[0]

    def learn(self, input_dir):
        """Learning model"""

        languages = self.languages
        extensions = [ext for exts in languages.values() for ext in exts]
        print (extensions)
        files = search_files(input_dir, extensions)
        nb_files = len(files)
        chunk_size = min(int(_CHUNK_PROPORTION * nb_files), _CHUNK_SIZE)

        batches = _pop_many(files, chunk_size)

        evaluation_data = extract_from_files(next(batches), languages)

        x_test = evaluation_data[0]
        y_test =  tf.keras.utils.to_categorical(
                evaluation_data[1], 
                num_classes=len(self.languages)
                )

        accuracy = 0
        total = ceil(nb_files / chunk_size) - 1
        print("Start learning")
        for pos, training_files in enumerate(batches, 1):

            training_data = extract_from_files(training_files, languages)
            
            x_train = training_data[0]
            y_train = tf.keras.utils.to_categorical(
                training_data[1], 
                num_classes=len(self.languages)
                )
            
            self._classifier.fit(
                x_train,
                y_train,
                batch_size = 64,
                epochs = 10,
                callbacks = [self.cp_callback])
            
            # evaluation
        accuracy = self._classifier.evaluate(
            x_test,
            y_test,
            verbose = 1)
        print("Accuracy: %.2f%%" % (accuracy[1]*100))

        return accuracy


    def learnPartial(self, input_dir):
        """Learning model"""

        languages = self.languages
        extensions = [ext for exts in languages.values() for ext in exts]
        print (extensions)
        files = search_files(input_dir, extensions)
        nb_files = len(files)
        chunk_size = min(int(_CHUNK_PROPORTION * nb_files), _CHUNK_SIZE)

        batches = _pop_many(files, chunk_size)

        evaluation_data = extract_from_files(next(batches), languages)

        x_test = evaluation_data[0]
        y_test =  tf.keras.utils.to_categorical(
                evaluation_data[1], 
                num_classes=len(self.languages)
                )

        accuracy = 0
        total = ceil(nb_files / chunk_size) - 1
        print("Start learning")
        for pos, training_files in enumerate(batches, 1):

            training_data = extract_from_files(training_files, languages)
            
            x_train = training_data[0]
            y_train = tf.keras.utils.to_categorical(
                training_data[1], 
                num_classes=len(self.languages)
                )
            
            self._classifier.load_weights(self.checkpoint_path)

            self._classifier.fit(
                x_train,
                y_train,
                batch_size = 64,
                epochs = 10,
                callbacks = [self.cp_callback])
            
            # evaluation
        accuracy = self._classifier.evaluate(
            x_test,
            y_test,
            verbose = 1)
        print("Accuracy: %.2f%%" % (accuracy[1]*100))

        return accuracy


def _pop_many(items, chunk_size):
    while items:
        yield items[0:chunk_size]

        # Avoid memory overflow
        del items[0:chunk_size]
        gc.collect()

def _to_func(vector):
    return lambda: (
        tf.constant(vector[0], name='const_features'),
        tf.constant(vector[1], name='const_labels'))
