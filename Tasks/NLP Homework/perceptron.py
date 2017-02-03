import tensorflow as tf
from collections import Counter
import math


f_trn = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/sst.trn.tsv"
f_dev = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/sst.dev.tsv"
f_tst = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/sst.tst.tsv"


def tfidf(f_input):
    term_f = []
    vocabulary = []
    with open(f_input, "rb") as f:
        for line in f:
            value = line.split("\t")
            label = value[0]
            doc = value[1].split()
            word_count_dic = Counter(doc)
            tf_doc = (label, word_count_dic)
            term_f.append(tf_doc)
            vocabulary = set(vocabulary.extend(word_count_dic.keys()))

    df_temp = []
    for ele in term_f:
        df_temp.extend(ele[1].keys())
    df = Counter(df_temp)

    D = len(term_f)

    tf_idf = []
    for doc in term_f:
        tf_idf_temp = {}
        for key in doc[1].keys():
            tf_idf_temp[key] = doc[1][key] * math.log(float(D) / float(df[key]))
        tf_idf.append((doc[0], tf_idf_temp))

    return tf_idf, vocabulary

def vector(base, tf_idf):
    vec_doc = []
    for ele in tf_idf:
        vec = len(base) * [0]
        label = ele[0]
        for key in ele[1].keys():
            vec[base.index[key]] = ele[1][key]
        vec_doc.append((label, vec))

    return vec_doc

class Perceptron(object):
    def __init__(self, learning_rate, epochs, batch_size, input_dim, num_classes):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.l2_lambda = 5e-4

    def model(self, x):
        W = tf.Variable(tf.truncated_normal([self.input_dim, self.num_classes]), "float", name="weights")
        b = tf.Variable(tf.zeros([self.num_classes]), "float", name="biases")

        y = tf.matmul(x, W) + b
        loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

        return y, loss_l2

    def train(self, x):
        x =
        predict, l2_loss = self.model(x)

tf_idf_trn, vocab_trn = tfidf(f_trn)
tf_idf_dev, vocab_dev = tfidf(f_dev)
tf_idf_tst, vocab_tst = tfidf(f_tst)

vocab = sorted(set(vocab_trn + vocab_dev + vocab_tst))
vec_len = len(vocab)
vec_trn = vector(vocab, tf_idf_trn)
vec_dev = vector(vocab, tf_idf_dev)
vec_tst = vector(vocab, tf_idf_tst)





