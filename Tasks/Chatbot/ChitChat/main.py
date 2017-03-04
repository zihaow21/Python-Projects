from data_utils import DataUtils
from NeuralNets.seq2seq_generative import Seq2seq
import pickle


# model_dir = '/home/zwan438/temp_folder/chitchat.ckpt'
# model_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt'
model_dir = '/home/zihao/temp_folder/chitchat.ckpt'

# meta_dir = '/home/zwan438/temp_folder/chitchat.ckpt.meta'
# meta_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt.meta'
meta_dir = '/home/zihao/temp_folder/chitchat.ckpt.meta'

# data_conversation_dir = "/home/zwan438/Dropbox/data/chitchat_conversation_data.txt"
# data_conversation_dir = "/Users/ZW/Dropbox/data/chitchat_conversation_data.txt"
data_conversation_dir = "/home/zihao/temp_folder/chit_chat_conversation.txt"

maxLength = 20
maxLengthEnco = 20
maxLengthDeco = 22
batchSize = 500
du = DataUtils(maxLength, maxLengthEnco, maxLengthDeco, batchSize, data_conversation_dir)
source_vocab_size = du.source_vocab_size
target_vocab_size = du.target_vocab_size

seq2seq = Seq2seq(epochs=1, learning_rate=0.0005, batch_size=batchSize, source_vocab_size=source_vocab_size,
                  target_vocab_size=target_vocab_size, maxLengthEnco=maxLengthEnco, maxLengthDeco=maxLengthEnco, num_softmax_samples=64,
                  embedding_size=100, hidden_size=64, num_layers=3, use_lstm=False, model_dir=model_dir,
                  meta_dir=meta_dir, num_threads=10, data_object=du)

seq2seq.train()


