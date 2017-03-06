import fasttext as fft
import pickle


class QueryClass(object):
    def __init__(self, conversation_data_dir, topic_data_dir, query_type_data_dir, conversation_index_dir):
        self.conversation_data_dir = conversation_data_dir
        self.topic_data_dir = topic_data_dir
        self.query_type_data_dir = query_type_data_dir
        self.conversation_index_dir = conversation_index_dir

        # self.conversation_classcifier = fft.supervised(self.conversation_data_dir, 'conversation_model', label_prefix='__label__', thread=3, epoch=10, ws=3)
        self.topic_classcifier = fft.supervised(self.topic_data_dir, 'topic_model', label_prefix='__label__', thread=1, epoch=100, ws=10)
        self.query_type_classcifier = fft.supervised(self.query_type_data_dir, 'query_type_model', label_prefix='__label__', thread=2, epoch=100, ws=10)

        # self.conversation_classcifier = fft.load_model('conversation_model', label_prefix='__label__')
        self.topic_classcifier = fft.load_model('topic_model.bin', label_prefix='__label__')
        self.query_type_classcifier = fft.load_model('query_type_model.bin', label_prefix='__label__')

        # self.query_lookup_source = pickle.load(self.conversation_index_dir)
        # self.utterance = None
        # self.label = None

    def queryTypeClass(self, query):
        self.label = self.query_type_classcifier.predict(query)

    def topicGenreClass(self, query):
        self.label = self.topic_classcifier.predict(query)

    def queryClassTest(self, query_test_dir):
        result = self.query_type_classcifier.test(query_test_dir)
        return result

    def topicClassTest(self, topic_test_dir):
        result = self.topic_classcifier.test(topic_test_dir)
        return result

    # def conversationClass(self, query):
    #     self.label = self.conversation_classcifier.predict(query)
    #
    # def classLookup(self):
    #     self.utterance = self.query_lookup_source[self.label]
    #     return self.utterance