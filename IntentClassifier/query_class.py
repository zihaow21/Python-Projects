import fasttext as fft
import pickle


class QueryClass(object):
    def __init__(self, conversation_data_dir, topic_data_dir, query_type_data_dir, conversation_index_dir):
        self.conversation_data_dir = conversation_data_dir
        self.topic_data_dir = topic_data_dir
        self.query_type_data_dir = query_type_data_dir
        self.conversation_index_dir = conversation_index_dir

        # self.conversation_classcifier = fft.supervised(self.conversation_data_dir, 'conversation_model', label_prefix='__label__')
        # self.topic_classcifier = fft.supervised(self.topic_data_dir, 'topic_model', label_prefix='__label__')
        # self.query_type_classcifier = fft.supervised(self.query_type_data_dir, 'query_type_model', label_prefix='__label__')

        self.conversation_classcifier = fft.load_model('conversation_model', label_prefix='__label__')
        self.topic_classcifier = fft.load_model('topic_model', label_prefix='__label__')
        self.query_type_classcifier = fft.load_model('query_type_model', label_prefix='__label__')

        self.query_lookup_source = pickle.load(self.conversation_index_dir)
        self.utterance = None
        self.label = None

    def queryTypeClass(self, query):
        self.label = self.query_type_classcifier.predict(query)

    def conversationClass(self, query):
        self.label = self.conversation_classcifier.predict(query)

    def classLookup(self):
        self.utterance = self.query_lookup_source[self.label]