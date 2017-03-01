from tqdm import tqdm
from cornell_data import CornellData
# from data_utils import DataUtils
from data_utils_word2vec import DataUtils
from NeuralNets.seq2seq_generative import Seq2seq
from gensim.models import Word2Vec as wv


movie_lines_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_lines.txt'
# movie_lines_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_lines.txt'
# movie_lines_filename = '/home/zihao/temp_folder/movie_lines.txt'

movie_conversations_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_conversations.txt'
# movie_conversations_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_conversations.txt'
# movie_conversations_filename = '/home/zihao/temp_folder/movie_conversations.txt'

# model_dir = '/home/zwan438/temp_folder/chitchat.ckpt'
model_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt'
# model_dir = '/home/zihao/temp_folder/chitchat.ckpt'

# meta_dir = '/home/zwan438/temp_folder/chitchat.ckpt.meta'
meta_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt.meta'
# meta_dir = '/home/zihao/temp_folder/chitchat.ckpt.meta'

# data_dir = '/home/zwan438/temp_folder/chitchat_data.txt'
data_sentence_dir = '/Users/ZW/Dropbox/Current/temp/chitchat_sentence_data.txt'
data_conversation_dir = '/Users/ZW/Dropbox/Current/temp/chitchat_conversation_data.txt'
# data_dir = '/home/zihao/temp_folder/chitchat_data.txt'

cd = CornellData(movie_lines_filename, movie_conversations_filename)
conversations = cd.getConversations()
# conversations = None

MAX_CONTENT_LENGTH = 30
MAX_UTTERANCE_LENGTH = 30
EMBEDDING_SIZE = 300
LSTM_CELL_SIZE = 256

du = DataUtils(conversations, MAX_CONTENT_LENGTH, MAX_CONTENT_LENGTH, MAX_UTTERANCE_LENGTH, 1000, data_conversation_dir)  # DataUtils(conversations, maxLength, maxLengthEnco, maxLengthDeco, batchSize)
du.createSentenceCorpus()
du.saveData()
# data = du.loadData()

# source_vocab_size = len(du.word2id)
# target_vocab_size = len(du.word2id)
#
# seq2seq = Seq2seq(epochs=1, learning_rate=0.0005, batch_size=1000, source_vocab_size=source_vocab_size,
#                   target_vocab_size=target_vocab_size, maxLengthEnco=MAX_CONTENT_LENGTH, maxLengthDeco=MAX_UTTERANCE_LENGTH + 2, num_softmax_samples=LSTM_CELL_SIZE,
#                   embedding_size=EMBEDDING_SIZE, hidden_size=LSTM_CELL_SIZE, num_layers=3, use_lstm=False, model_dir=model_dir,
#                   meta_dir=meta_dir, num_threads=10, data_object=du)
#
# # seq2seq.train()
# # with open("/Users/ZW/Downloads/cornell movie-dialogs corpus/samples.txt") as f:
# # with open("/home/zihao/temp_folder/samples.txt") as f:
# with open("/home/zwan438/temp_folder/samples.txt") as f:
#     questions = f.readlines()
#     answers = seq2seq.generation(questions)
#     for i in range(len(answers)):
#         print "Question: {}. Answer: {}".format(questions[i], answers[i])

