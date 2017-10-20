import os
import sys
import pandas as pd

base_path = './'

def normal_clean():
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

    questions = []
    tags = []
    idxs = []

    for idx, line in enumerate(test_lines):
        question = line.split(',,,')[0]
        tag = line.split(',,,')[1].strip('\n').strip(' ')
        questions.append(question)
        tags.append(tag)
        idxs.append(idx)

    df = pd.DataFrame(columns=columns)
    df['question'] = questions
    df['tag'] = tags
    df['id'] = idxs
    df.to_pickle(base_path+'TestingDataPickle.pkl')

def web_data_clean():
    training_data_file = base_path + 'WebTestData.txt'

    f = open(training_data_file, 'r')

    all_lines = f.readlines()

    columns = ['id', 'question', 'tag']

    questions = []
    tags = []
    idxs = []

    for idx, line in enumerate(all_lines):
        tag = line.split(':')[0]
        question = line.split(':')[1].strip('\n').strip(' ')
        questions.append(question)
        tags.append(tag)
        idxs.append(idx)

    df = pd.DataFrame(columns=columns)
    df['question'] = questions
    df['tag'] = tags
    df['id'] = idxs
    df.to_pickle(base_path+'WebTestDataPickle.pkl')

if __name__ == '__main__':
    normal_clean()
    # web_data_clean()
