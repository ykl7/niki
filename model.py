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
        pass

    def fit_model(self, inputs, outputs, epochs):
        pass