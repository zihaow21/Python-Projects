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
"""

query_type_fileName = '/Users/ZW/Dropbox/data/query_type_train.txt'

genre_list = ['what', 'why', 'when', 'who', 'where', 'how', 'do', 'which']
genres = dict()
for i, g in enumerate(genre_list):
    genres[g] = i

conprehensive_data_fileName = '/Users/ZW/baidu/data/selqa_ass/SelQA-ass-train_analysis.pickle'
with open(conprehensive_data_fileName, 'rb') as f:
    data = pickle.load(f)

utterances = []
for data_item in data:
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

with open(query_type_fileName, 'wb') as f:
    for utterance in utterances:
        print >> f, utterance

    f.close()
