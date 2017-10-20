import os
import sys
import pickle
import keras
import json

from datetime import datetime
import pandas as pd
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout, Flatten
from keras.layers.merge import concatenate, dot, multiply
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from data_clean import web_data_clean

base_path = './'

class Detector:

    def __init__(self, question_max, word_embedding_size, doc2vec_size):
        self.model = None
        self.question_max = question_max
        self.word_embedding_size = word_embedding_size
        self.doc2vec_size = doc2vec_size

    def set_params(self, activation):
        self.activation = activation

    def create_model(self):
        question_words = Input(shape=(self.question_max, self.word_embedding_size))

        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(question_words)
        flattener = Flatten()(lstm_layer)
        output = Dense(5, activation='sigmoid')(flattener)

        self.model = Model(inputs=[question_words], outputs=output)
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self, inputs, outputs, epochs):
        filepath = './weights/weights-{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, outputs, validation_split=0.2, epochs=epochs, callbacks=callbacks_list, verbose=1)

def train():
    training_pickle_name = base_path + 'TrainingDataPickle.pkl'
    df = pd.read_pickle(training_pickle_name)

    max_len = 30

    word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    words = []
    for index, row in df.iterrows():
        l = len(row['question'])
        temp = []
        if l >= max_len:
            for i in range(max_len):
                try:
                    temp.append(word_vectors[row['question'][i]])
                except:
                    temp.append(np.zeros(300))
        else:
            pad = max_len - l
            for i in range(pad):
                temp.append(np.zeros(300))
            for i in range(l):
                try:
                    temp.append(word_vectors[row['question'][i]])
                except:
                    temp.append(np.zeros(300))
        words.append(temp)

    words = np.array(words)
    tester = Detector(max_len, 300, 300)
    tester.set_params('relu')
    tester.create_model()

    encoder = LabelEncoder()
    tags = df['tag']
    encoder.fit(tags)
    encoded_Y = encoder.transform(tags)
    dummy_y = np_utils.to_categorical(encoded_Y)

    tester.fit_model(words, dummy_y, 10)

def internal_test():
    training_pickle_name = base_path + 'TrainingDataPickle.pkl'
    df = pd.read_pickle(training_pickle_name)

    max_len = 30

    word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    words = []
    for index, row in df.iterrows():
        l = len(row['question'])
        temp = []
        if l >= max_len:
            for i in range(max_len):
                try:
                    temp.append(word_vectors[row['question'][i]])
                except:
                    temp.append(np.zeros(300))
        else:
            pad = max_len - l
            for i in range(pad):
                temp.append(np.zeros(300))
            for i in range(l):
                try:
                    temp.append(word_vectors[row['question'][i]])
                except:
                    temp.append(np.zeros(300))
        words.append(temp)

    words = np.array(words)
    tester = Detector(max_len, 300, 300)
    tester.set_params('relu')
    tester.create_model()
    tester.model.load_weights('./weights/weights-08-0.65.hdf5')
    out = tester.model.predict(words)

def test():
    web_data_clean()
    training_pickle_name = base_path + 'WebTestDataPickle.pkl'
    df = pd.read_pickle(training_pickle_name)

    max_len = 30

    word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    words = []
    for index, row in df.iterrows():
        l = len(row['question'])
        temp = []
        if l >= max_len:
            for i in range(max_len):
                try:
                    temp.append(word_vectors[row['question'][i]])
                except:
                    temp.append(np.zeros(300))
        else:
            pad = max_len - l
            for i in range(pad):
                temp.append(np.zeros(300))
            for i in range(l):
                try:
                    temp.append(word_vectors[row['question'][i]])
                except:
                    temp.append(np.zeros(300))
        words.append(temp)

    words = np.array(words)
    tester = Detector(max_len, 300, 300)
    tester.set_params('relu')
    tester.create_model()
    tester.model.load_weights('./weights/weights-08-0.65.hdf5')
    out = tester.model.predict(words)

if __name__ == '__main__':
    # internal_test()
    test()

