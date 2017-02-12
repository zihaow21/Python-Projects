from __future__ import division
from collections import Counter
import math
import random
import numpy as np
import pickle
from tqdm import tqdm


f_trn = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/sst.trn.tsv"
f_dev = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/sst.dev.tsv"
f_tst = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/sst.tst.tsv"

model_dir_tfidf = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/perceptron_tfidf.pik"
model_dir_onehot = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework3/perceptron_onehot.pik"

#  tfidf method
def tfidf(f_input):
    term_f = []
    vocabulary = []
    with open(f_input, "rb") as f:
        for line in f:
            value = line.split("\t")
            label = int(value[0])
            doc = value[1].split()
            word_count_dic = Counter(doc)
            tf_doc = (label, word_count_dic)
            term_f.append(tf_doc)
            key_list = word_count_dic.keys()
            vocabulary.extend(key_list)
        vocabulary = set(vocabulary)

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


def tfidf_vector(base, tf_idf):
    vec_doc = []
    for ele in tf_idf:
        vec = len(base) * [0]
        label = ele[0]
        for key in ele[1].keys():
            index = base.index(key)
            vec[index] = ele[1][key]
        vec_doc.append((label, vec))

    return vec_doc


# oneHot method
def oneHot_vocab(f_input):
    vocab = []
    doc_tuple = []
    with open(f_input, "rb") as f:
        for line in f:
            value = line.split("\t")
            label = int(value[0])
            doc = list(set(value[1].split()))
            doc_tuple.append((label, doc))
            vocab.extend(doc)
    vocabulary = sorted(list(set(vocab)))

    return vocabulary, doc_tuple

def onehot_vector(base, doc_tuple):
    onehot_vec = []
    for label, vocab in doc_tuple:
        vec_temp = [0] * len(base)
        for v in vocab:
            vec_temp[base.index(v)] = 1
        onehot_vec.append((label, vec_temp))

    return onehot_vec

#  tfidf method
tf_idf_trn, vocab_trn = tfidf(f_trn)
tf_idf_dev, vocab_dev = tfidf(f_dev)
tf_idf_tst, vocab_tst = tfidf(f_tst)

vocab = sorted(list(set(list(vocab_trn) + list(vocab_dev) + list(vocab_tst))))
vec_len = len(vocab)
vec_trn = tfidf_vector(vocab, tf_idf_trn)
vec_dev = tfidf_vector(vocab, tf_idf_dev)
vec_tst = tfidf_vector(vocab, tf_idf_tst)

#  onehot method
vocabulary_trn, doc_tuple_trn = oneHot_vocab(f_trn)
vocabulary_dev, doc_tuple_dev = oneHot_vocab(f_dev)
vocabulary_tst, doc_tuple_tst = oneHot_vocab(f_tst)

vocabulary = sorted(list(set(list(vocabulary_trn) + list(vocabulary_dev) + list(vocabulary_tst))))
vector_trn = onehot_vector(vocabulary, doc_tuple_trn)
vector_dev = onehot_vector(vocabulary, doc_tuple_dev)
vector_tst = onehot_vector(vocabulary, doc_tuple_tst)

class MultiClassPerceptron(object):
    def __init__(self, classes, data_train, epochs, model_dir):
        self.classes = classes
        self.data_train = data_train
        self.epochs = epochs
        self.model_dir = model_dir

        random.shuffle(self.data_train)
        self.weights = {cls: np.array([0.0 for _ in xrange(len(self.data_train[0][1]) + 1)]) for cls in self.classes}

    def train(self):
        for k in tqdm(xrange(self.epochs)):
            class_pair = []
            count = 0
            for label, vec in self.data_train:
                data = vec + [1.0]
                data = np.array(data)

                argmax, prediction = 0, self.classes[0]

                for cls in self.classes:
                    current = np.dot(data, self.weights[cls])

                    if current >= argmax:
                        argmax, prediction = current, cls

                class_pair.append([label, prediction])

                if label != prediction:
                    self.weights[label] += data
                    self.weights[prediction] -= data

            for i in xrange(len(class_pair)):
                if class_pair[i][0] == class_pair[i][1]:
                    count += 1

            accuracy = count/len(class_pair)

            if k % 5 == 0:
                print "The training accuracy for epoch {} is {}".format(k, accuracy)

    def saveModel(self):
        # with open(model_dir_tfidf, "wb") as f:
        with open(model_dir_onehot, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def loadModel(self):
        # with open(self.model_dir, "rb") as f:
        with open(model_dir_onehot, "rb") as f:
            return pickle.load(f)

    def pred(self, data_dev):
        class_pair = []
        count = 0
        for label, vec in data_dev:
            data = np.array(vec + [1.0])

            argmax, prediction = 0, self.classes[0]

            for cls in self.classes:
                current = np.dot(data, self.weights[cls])
                if current >= argmax:
                    argmax, prediction = current, cls

                class_pair.append([label, prediction])

        for i in xrange(len(class_pair)):
            if class_pair[i][0] == class_pair[i][1]:
                count += 1

        accuracy = count/len(class_pair)

        print accuracy

# mcp = MultiClassPerceptron([0, 1, 2, 3, 4], vec_trn, 15, model_dir_tfidf)
mcp = MultiClassPerceptron([0, 1, 2, 3, 4], vector_trn, 33, model_dir_onehot)
mcp.train()
mcp.saveModel()
mcp.loadModel()
# mcp.pred(vec_dev)
mcp.pred(vector_dev)