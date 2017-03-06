from fastText import FastText
from query_class import QueryClass


# model_dir = '/Users/ZW/Dropbox/current/Python-Projects/IntentClassifier/fast_text/model.bin'
# ftt = FastText(model_dir)
# print ftt.modelCheck('king')

conversation_data_dir = '/Users/ZW/Dropbox/data/conversation_train.txt'
topic_data_dir = '/Users/ZW/Dropbox/data/topic_train.txt'
query_type_data_dir ='/Users/ZW/Dropbox/data/query_type_train.txt'
conversation_index_dir = '/Users/ZW/Dropbox/data/conversation_index.txt'

qc = QueryClass(conversation_data_dir=conversation_data_dir, topic_data_dir=topic_data_dir,
                query_type_data_dir=query_type_data_dir, conversation_index_dir=conversation_index_dir)

query = 'How are you?'
qc.conversationClass(query)
qc.classLookup()
print qc.utterance