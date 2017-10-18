import os
import sys
import pickle
import keras
import json

from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout
from keras.layers.merge import concatenate, dot, multiply
from keras.models import Model
from keras.callbacks import ModelCheckpoint

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
        question_embed_input = Input(shape=(self.doc2vec_size, ))

        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(question_words)
        output = Dense(1, activation='sigmoid')(lstm_layer)

        self.model = Model(inputs=[question_words] + [question_embed_input], outputs=output)
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self, inputs, outputs, epochs):
        filepath = './weights-{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, outputs, validation_split=0.2, epochs=epochs, callbacks=callbacks_list, verbose=1)

