import os
import sys
import pandas as pd

base_path = './'

training_data_file = base_path + 'TrainingData.txt'

f = open(training_data_file, 'r')

all_lines = f.readlines()

train_lines = all_lines[:int(len(all_lines)*0.7)]
test_lines = all_lines[int(len(all_lines)*0.7):]

columns = ['id', 'question', 'tag']

questions = []
tags = []
idxs = []

for idx, line in enumerate(train_lines):
    question = line.split(',,,')[0]
    tag = line.split(',,,')[1].strip('\n').strip(' ')
    questions.append(question)
    tags.append(tag)
    idxs.append(idx)

df = pd.DataFrame(columns=columns)
df['question'] = questions
df['tag'] = tags
df['id'] = idxs
df.to_pickle(base_path+'TrainingDataPickle.pkl')

