from fastText import FastText
from query_class import QueryClass


# model_dir = '/Users/ZW/Dropbox/current/Python-Projects/IntentClassifier/fast_text/model.bin'
# ftt = FastText(model_dir)
# print ftt.modelCheck('king')

# conversation_data_dir = '/Users/ZW/Dropbox/data/conversation_train.txt'
# topic_data_dir = '/Users/ZW/Dropbox/data/topic_train.txt'
# query_type_data_dir ='/Users/ZW/Dropbox/data/query_type_train.txt'
# conversation_index_dir = '/Users/ZW/Dropbox/data/conversation_index.txt'

conversation_data_dir = '/home/zwan438/Dropbox/data/conversation_train.txt'
topic_data_dir = '/home/zwan438/Dropbox/data/topic_train.txt'
query_type_data_dir ='/home/zwan438/Dropbox/data/query_type_train.txt'
conversation_index_dir = '/home/zwan438/Dropbox/data/conversation_index.txt'

query_test_dir = '/home/zwan438/Dropbox/data/query_type_test.txt'
topic_test_dir = '/home/zwan438/Dropbox/data/topic_test.txt'

qc = QueryClass(conversation_data_dir=conversation_data_dir, topic_data_dir=topic_data_dir,
                query_type_data_dir=query_type_data_dir, conversation_index_dir=conversation_index_dir)

result_query = qc.queryClassTest(query_test_dir)
print 'The query class prediction precision is {} \n'.format(result_query.precision)
print 'The query class prediction recall is {} \n'.format(result_query.recall)
print 'The query class prediction samples are {} \n'.format(result_query.nexamples)

result_topic = qc.topicClassTest(topic_test_dir)
print 'The topic class prediction precision is {} \n'.format(result_topic.precision)
print 'The topic class prediction recall is {} \n'.format(result_topic.recall)
print 'The topic class prediction samples are {} \n'.format(result_topic.nexamples)

# query = 'How are you?'
# qc.conversationClass(query)
# qc.classLookup()
# print qc.utterance