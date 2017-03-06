import pickle
import nltk


"""
Type indices:
what = 0
why = 1
when = 2
who = 3
where = 4
how = 5
do = 6
which = 7
statement = 8
"""

query_type_train_fileName = '/home/zwan438/Dropbox/data/query_type_train.txt'
query_type_test_fileName = '/home/zwan438/Dropbox/data/query_type_test.txt'
genre_list = ['what', 'why', 'when', 'who', 'where', 'how', 'do', 'which', 'statement']
genres = dict()
for i, g in enumerate(genre_list):
    genres[g] = i

conprehensive_data_train_fileName = '/home/zwan438/Dropbox/data/SelQA-ass-train_analysis.pickle'
conprehensive_data_test_fileName = '/home/zwan438/Dropbox/data/SelQA-ass-test_analysis.pickle'
question_answer_train_fileName = '/home/zwan438/Dropbox/data/SelQA-ass-train.txt'
question_answer_test_fileName = '/home/zwan438/Dropbox/data/SelQA-ass-test.txt'

with open(conprehensive_data_train_fileName, 'rb') as f:
    train_data = pickle.load(f)

with open(conprehensive_data_test_fileName, 'rb') as f:
    test_data = pickle.load(f)

with open(question_answer_train_fileName, 'rb') as f:
    train_data_statement = []
    data = f.readlines()
    for line in data:
        line_split = line.split('\t')
        train_data_statement.append(line_split)

with open(question_answer_test_fileName, 'rb') as f:
    test_data_statement = []
    data = f.readlines()
    for line in data:
        line_split = line.split('\t')
        test_data_statement.append(line_split)

utterances = []
for data_item in train_data:
    if len(data_item['q_types']) == 0:
        tokens = nltk.word_tokenize(data_item['question'])
        if 'which' in tokens or 'Which' in tokens:
            label = genres['which']
        else:
            label = genres['do']
    else:
        label = genres[next(iter(data_item['q_types']))]
    question = ''.join(['__label__', str(label), ' ,', data_item['question'], '\n']).encode('utf-8').strip()
    utterances.append(question)

for data_item in train_data_statement:
    label = genres['statement']
    statement = ''.join(['__label__', str(label), ' ,', data_item[1], '\n'])
    utterances.append(statement)

with open(query_type_train_fileName, 'wb') as f:
    for utterance in utterances:
        print >> f, utterance

    f.close()

utterances = []
for data_item in test_data:
    if len(data_item['q_types']) == 0:
        tokens = nltk.word_tokenize(data_item['question'])
        if 'which' in tokens or 'Which' in tokens:
            label = genres['which']
        else:
            label = genres['do']
    else:
        label = genres[next(iter(data_item['q_types']))]
    question = ''.join(['__label__', str(label), ' ,', data_item['question'], '\n']).encode('utf-8').strip()
    utterances.append(question)

for data_item in test_data_statement:
    label = genres['statement']
    statement = ''.join(['__label__', str(label), ' ,', data_item[1], '\n'])
    utterances.append(statement)

with open(query_type_test_fileName, 'wb') as f:
    for utterance in utterances:
        print >> f, utterance

    f.close()
