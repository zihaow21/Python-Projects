from tqdm import tqdm
from cornell_data import CornellData
from data_utils import DataUtils
from NeuralNets.seq2seq_generative import Seq2seq


# movie_lines_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_lines.txt'
movie_lines_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_lines.txt'
# movie_conversations_filename = '/Users/ZW/Downloads/cornell movie-dialogs corpus/movie_conversations.txt'
movie_conversations_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_conversations.txt'

model_dir = '/home/zwan438/temp_folder/chitchat.ckpt'
# model_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt'
meta_dir = '/home/zwan438/temp_folder/chitchat.meta'
# meta_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.meta'

data_dir = '/home/zwan438/temp_folder/chitchat_data.txt'
# data_dir = '/Users/ZW/Dropbox/Current/temp/chitchat_data.txt'


# cd = CornellData(movie_lines_filename, movie_conversations_filename)
# conversations = cd.getConversations()
conversations = None
du = DataUtils(conversations, 20, 20, 22, 100, data_dir)  # DataUtils(conversations, maxLength, maxLengthEnco, maxLengthDeco, batchSize)
# du.createCorpus()
# du.saveData()

du.loadData()
source_vocab_size = len(du.word2id)
target_vocab_size = len(du.word2id)

seq2seq = Seq2seq(epochs=10000, learning_rate=0.005, batch_size=100, source_vocab_size=source_vocab_size,
                  target_vocab_size=target_vocab_size, maxLengthEnco=20, maxLengthDeco=22, num_softmax_samples=512,
                  embedding_size=50, hidden_size=512, num_layers=3, use_lstm=False, model_dir=model_dir,
                  meta_dir=meta_dir, num_threads=50)

epochs = 11
for epoch in tqdm(range(epochs)):
    batches = du.getBatches()
    for batch in batches:
        seq2seq.train(batch, epoch)


