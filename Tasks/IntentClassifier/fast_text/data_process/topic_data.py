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
POLITICS = 10
"""

topic_train_fileName = '/home/zwan438/Dropbox/data/topic_train.txt'
topic_test_fileName = '/home/zwan438/Dropbox/data/topic_test.txt'

genre_list = ['MUSIC', 'TV', 'TRAVEL', 'ART', 'SPORT', 'COUNTRY', 'MOVIES', 'HISTORICAL EVENTS', 'SCIENCE', 'FOOD', 'POLITICS']
genres = dict()
for i, g in enumerate(genre_list):
    genres[g] = i

conprehensive_data_train_fileName = '/home/zwan438/Dropbox/data/SelQA-ass-train_raw.pickle'
conprehensive_data_test_fileName = '/home/zwan438/Dropbox/data/SelQA-ass-test_raw.pickle'

political_train_fileName  = '/home/zwan438/Dropbox/data/political_train_utterance.txt'
political_test_fileName  = '/home/zwan438/Dropbox/data/political_test_utterance.txt'

with open(conprehensive_data_train_fileName, 'rb') as f:
    train_data_question = pickle.load(f)
with open(political_train_fileName, 'rb') as f:
    political_train_data = f.readlines()

with open(conprehensive_data_train_fileName, 'rb') as f:
    test_data_question = pickle.load(f)
with open(political_test_fileName, 'rb') as f:
    political_test_data = f.readlines()

utterances = []
for data_item in train_data_question:
    label = genres[data_item['type']]
    question = ''.join(['__label__', str(label), ' ,', data_item['question'], '\n']).encode('utf-8').strip()
    utterances.append(question)
    for c in data_item['candidates']:
        answer = ''.join(['__label__', str(label), ' ,', data_item['sentences'][c], '\n']).encode('utf-8').strip()
        utterances.append(answer)
for data_item in political_train_data:
    label = genres['POLITICS']
    statement = ''.join(['__label__', str(label), ' ,', data_item, '\n']).encode('utf-8').strip()
    utterances.append(statement)

with open(topic_train_fileName, 'wb') as f:
    for utterance in utterances:
        print >> f, utterance

    f.close()


utterances = []
for data_item in test_data_question:
    label = genres[data_item['type']]
    question = ''.join(['__label__', str(label), ' ,', data_item['question'], '\n']).encode('utf-8').strip()
    utterances.append(question)
    for c in data_item['candidates']:
        answer = ''.join(['__label__', str(label), ' ,', data_item['sentences'][c], '\n']).encode('utf-8').strip()
        utterances.append(answer)
for data_item in political_test_data:
    label = genres['POLITICS']
    statement = ''.join(['__label__', str(label), ' ,', data_item, '\n']).encode('utf-8').strip()
    utterances.append(statement)
with open(topic_test_fileName, 'wb') as f:
    for utterance in utterances:
        print >> f, utterance

    f.close()

