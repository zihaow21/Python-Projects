import pickle


"""
Type indices:
MUSIC = 0
TV = 1
TRAVEL = 2
ART = 3
SPORT = 4
COUNTRY = 5
MOVIES = 6
HISTORICAL EVENTS = 7
SCIENCE = 8
FOOD = 9
"""

topic_train_fileName = '/Users/ZW/Dropbox/data/topic_train.txt'

genre_list = ['MUSIC', 'TV', 'TRAVEL', 'ART', 'SPORT', 'COUNTRY', 'MOVIES', 'HISTORICAL EVENTS', 'SCIENCE', 'FOOD']
genres = dict()
for i, g in enumerate(genre_list):
    genres[g] = i

conprehensive_data_fileName = '/Users/ZW/baidu/data/selqa_ass/SelQA-ass-train_raw.pickle'
with open(conprehensive_data_fileName, 'rb') as f:
    data = pickle.load(f)

utterances = []
for data_item in data:
    label = genres[data_item['type']]
    question = ''.join(['__label__', str(label), ' ,', data_item['question'], '\n']).encode('utf-8').strip()
    utterances.append(question)
    for c in data_item['candidates']:
        answer = ''.join(['__label__', str(label), ' ,', data_item['sentences'][c], '\n']).encode('utf-8').strip()
        utterances.append(answer)
with open(topic_train_fileName, 'wb') as f:
    for utterance in utterances:
        print >> f, utterance

    f.close()
