from data_utils import DataUtils
import pickle


# model_dir = '/home/zwan438/temp_folder/chitchat.ckpt'
model_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt'
# model_dir = '/home/zihao/temp_folder/chitchat.ckpt'

# meta_dir = '/home/zwan438/temp_folder/chitchat.ckpt.meta'
meta_dir = '/Users/ZW/Dropbox/Current/temp/chitchat.ckpt.meta'
# meta_dir = '/home/zihao/temp_folder/chitchat.ckpt.meta'

# word2vec_index_dir = "/home/zwan438/Dropbox/data/word2vec_index.txt"
word2vec_index_dir = "/Users/ZW/Dropbox/data/word2vec_index.txt"
# word2vec_index_dir = "/home/zihao/Dropbox/data/word2vec_index.txt"

# data_conversation_dir = "/home/zwan438/Dropbox/data/word2vec_index.txt"
data_conversation_dir = "/Users/ZW/Dropbox/data/chitchat_conversation_data.txt"
# data_conversation_dir = "/home/zihao/Dropbox/data/word2vec_index.txt"

maxLength = 40
maxLengthEnco = 40
maxLengthDeco = 22
batchSize = 1000
du = DataUtils(maxLength, maxLengthEnco, maxLengthDeco, batchSize, word2vec_index_dir, data_conversation_dir)

with open(data_conversation_dir, 'r') as f:
    data = pickle.load(data_conversation_dir)
    sample = data['trainingSample']
    word2id = data['word2id']
    id2word = data['id2word']


