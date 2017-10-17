import os
import sys
import json
import pickle

import pandas as pd

from gensim.models import Doc2Vec, doc2vec

base_path = '/Users/apple/Documents/Python/niki/'

class Embedding():

    def __init__(self, questions, questionIds):
        self.questions = questions
        self.questionIds = questionIds
        self.labelledQuestions = []

    def label(self):
        for i in range(len(self.questions)):
            self.labelledQuestions.append(doc2vec.LabeledSentence(words=self.questions[i].split(), tags=['question_%s' % self.questionIds[i]]))

    def train(self):
        self.model = Doc2Vec()
        self.model.build_vocab(self.labelledQuestions)
        for i in range(10):
            self.model.train(self.labelledQuestions)

if __name__ == '__main__':
    training_pickle_name = base_path + 'TrainingDataPickle.pkl'
    df = pd.read_pickle(training_pickle_name)

    e = Embedding(df['question'], df['id'])
    e.label()
    e.train()
    e.model.save('./embed_model')

    question_embed = {}

    for k in range(len(df['id'])):
        question_embed['question_%s' % k] = e.model.docvecs['question_%s' % k]

    fp = open('./question_embed.pkl', 'w+')
    pickle.dump(question_embed, fp)

