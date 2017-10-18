import os
import sys
import pandas as pd

base_path = '/Users/apple/Documents/Python/niki/'

training_data_file = base_path + 'TrainingData.txt'

f = open(training_data_file, 'r')

all_lines = f.readlines()

train_lines = all_lines[:int(len(all_lines)*0.7)]
test_lines = all_lines[int(len(all_lines)*0.7):]

columns = ['id', 'question', 'type', 'output']

questions = []
tags = []
idxs = []
outputs = []

for idx, line in enumerate(train_lines):
    question = line.split(',,,')[0]
    tag = line.split(',,,')[1].strip('\n').strip(' ')
    if tag == 'what':
    	outputs.append(1)
    elif tag == 'who':
    	outputs.append(2)
    elif tag == 'when':
    	outputs.append(3)
    elif tag == 'affirmation':
    	outputs.append(4)
    elif tag == 'unknown':
    	outputs.append(5)
    questions.append(question)
    tags.append(tag)
    idxs.append(idx)

df = pd.DataFrame(columns=columns)
df['question'] = questions
df['tags'] = tags
df['id'] = idxs
df['output'] = outputs
df.to_pickle(base_path+'TrainingDataPickle.pkl')