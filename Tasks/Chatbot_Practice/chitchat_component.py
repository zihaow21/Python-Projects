from tqdm import tqdm
from cornell_data import CornellData
from data_utils import DataUtils
from NeuralNets.seq2seq_generative import Seq2seq


class ChitChat(object):
    def __init__(self):

        self.movie_lines_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_lines.txt'
        # self.movie_lines_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_lines.txt'
        # movie_lines_filename = '/home/zihao/temp_folder/movie_lines.txt'

        self.movie_conversations_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_conversations.txt'
        # self.movie_conversations_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_conversations.txt'
        # self.movie_conversations_filename = '/home/zihao/temp_folder/movie_conversations.txt'

        # self.model_dir = '/home/zwan438/temp_folder/chitchat.ckpt'
        self.model_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt'
        # self.model_dir = '/home/zihao/temp_folder/chitchat.ckpt'

        # self.meta_dir = '/home/zwan438/temp_folder/chitchat.meta'
        self.meta_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt.meta'
        # self.meta_dir = '/home/zihao/temp_folder/chitchat.meta'

        # self.data_dir = '/home/zwan438/temp_folder/chitchat_data.txt'
        self.data_dir = '/Users/ZW/Dropbox/Current/temp/chitchat_data.txt'
        # self.data_dir = '/home/zihao/temp_folder/chitchat_data.txt'

        self.conversations = None
        self.du = DataUtils(self.conversations, 10, 10, 12, 500, self.data_dir)  # DataUtils(conversations, maxLength, maxLengthEnco, maxLengthDeco, batchSize)
        self.du.loadData()
        self.source_vocab_size = len(self.du.word2id)
        self.target_vocab_size = len(self.du.word2id)

        self.seq2seq = Seq2seq(epochs=100, learning_rate=0.005, batch_size=500, source_vocab_size=self.source_vocab_size,
                               target_vocab_size=self.target_vocab_size, maxLengthEnco=10, maxLengthDeco=12,
                               num_softmax_samples=64,
                               embedding_size=100, hidden_size=64, num_layers=3, use_lstm=False,
                               model_dir=self.model_dir,
                               meta_dir=self.meta_dir, num_threads=10, data_object=self.du)

    def createData(self):
        cd = CornellData(self.movie_lines_filename, self.movie_conversations_filename)
        self.conversations = cd.getConversations()

    def mainTrain(self):
        self.du.createCorpus()
        self.du.saveData()
        self.seq2seq.train()

    def mainTest(self, question):
        #answer_code = self.seq2seq.generation(question)
        #answer_string = self.du.vec2str(answer_code)

        #return answer_string
        return "hi"






