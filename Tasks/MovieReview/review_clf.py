from FeatureExtraction.sent_preprocess import Sentence
from FeatureExtraction.sent_padding import Padding
from NeuralNets.rnn_multi_model import RNN_MultiModel_Dynamic
import numpy as np
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import sys
import io
import pickle


# ### preprocess sentences
# pos_file = "/Users/ZW/Downloads/pos-neg/pos.txt"
# neg_file = "/Users/ZW/Downloads/pos-neg/neg.txt"
# sent = Sentence()
# sent.sentence(pos_file, neg_file)

### load data
with io.open('/Users/ZW/Dropbox/current/temp/movie_sent.pickle', 'rb') as fs:
    sent = pickle.load(fs)
with io.open('/Users/ZW/Dropbox/current/temp/movie_label.pickle', 'rb') as fl:
    label = pickle.load(fl)
with io.open('/Users/ZW/Dropbox/current/temp/movie_vocab.pickle', 'rb') as fv:
    vocab = pickle.load(fv)
with io.open('/Users/ZW/Dropbox/current/temp/movie_vocab_No.pickle', 'rb') as fn:
    vocab_No = pickle.load(fn)
with io.open('/Users/ZW/Dropbox/current/temp/movie_seq_len.pickle', 'rb') as fg:
    seq_len = pickle.load(fg)
with io.open('/Users/ZW/Dropbox/current/temp/movie_sent_No.pickle', 'rb') as fsn:
    sent_No = pickle.load(fsn)

### padding sentences
pad = Padding()
data = pad.padding(sent_No, seq_len)

### get word embeddings
embedding_size = 128
vocab_size = len(vocab_No)
sequence_length = max(seq_len)

model_dir_rnn_multi = '/Users/ZW/Dropbox/current/temp/movie_review_rnn_multi.ckpt'
meta_dir_rnn_multi = '/Users/ZW/Dropbox/current/temp/movie_review_rnn_multi.ckpt.meta'

n_splits = 1
test_size = 0.25
ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

num_samples = np.int(len(data) * (1 - test_size))

data = np.array(data)
label = np.array(label)
for train_index, test_index in ss.split(data):
    data_train = data[train_index, :]
    label_train = label[train_index, :]
    data_test = data[test_index, :]
    label_test = label[test_index, :]

    rnn = RNN_MultiModel_Dynamic(epochs=300, num_samples=num_samples, learning_rate=0.1, batch_size=100,
                                 num_classes=2, num_steps= sequence_length, vocab_size=vocab_size,
                                 embedding_size=embedding_size, state_size=128, num_layers=3, model_dir=model_dir_rnn_multi,
                                 meta_dir=meta_dir_rnn_multi, type='GRU')

    rnn.classify_train_test(data_train, label_train, data_test, label_test)
