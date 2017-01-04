from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import io
import re
import pickle
from collections import Counter


class Sentence(object):
    def clean_str(self, string):
        string = re.sub(r"\'s", " s", string)
        string = re.sub(r"\'ve", " ve", string)
        string = re.sub(r"\'t", " t", string)
        string = re.sub(r"\'re", " re", string)
        string = re.sub(r"\'d", " d", string)
        string = re.sub(r"\'ll", " ll", string)
        string = re.sub(r'"', " ", string)
        string = re.sub(r'-', " ", string)

        return string

    def sentence(self, pos_file, neg_file):
        sent = []
        label = []
        dict = []

        ### stopwords and punctuation initialization
        stop_punc = stopwords.words('english') + list(string.punctuation)

        ### lemmatization initialization
        lemmatizer = WordNetLemmatizer()

        ### create lexicon
        with io.open(pos_file, 'r', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = self.clean_str(line)
                words = word_tokenize(line.lower())
                ### remove stopwords and punctuation
                temp = [lemmatizer.lemmatize(word, 'v') for word in words if word not in stop_punc]
                sent.append(temp)
                label.append([1, 0])
                dict += temp

        with io.open(neg_file, 'r', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = self.clean_str(line)
                words = word_tokenize(line.lower())
                ### remove stopwords and punctuation
                temp = [lemmatizer.lemmatize(word, 'v') for word in words if word not in stop_punc]
                sent.append(temp)
                label.append([0, 1])
                dict += temp

        dict_uniq = Counter(dict)
        dict_No = {}
        i = 0

        for word in dict_uniq.keys():
            if word not in dict_No:
                dict_No[word] = i
                i += 1

        seq = []
        for i in range(len(sent)):
            seq.append(len(sent[i]))

        ### look up a word's sequential number in the vocabulary to reform a sentence of numbers
        sent_No = []
        for i in range(len(sent)):
            temp = []
            for word in sent[i]:
                temp.append(dict_No[word])
            sent_No.append(temp)

        with io.open('/Users/ZW/Dropbox/current/temp/movie_sent.pickle', 'wb') as fs:
            pickle.dump(sent, fs)

        with io.open('/Users/ZW/Dropbox/current/temp/movie_label.pickle', 'wb') as fl:
            pickle.dump(label, fl)

        with io.open('/Users/ZW/Dropbox/current/temp/movie_vocab.pickle', 'wb') as fv:
            pickle.dump(dict_uniq, fv)

        with io.open('/Users/ZW/Dropbox/current/temp/movie_vocab_No.pickle', 'wb') as fn:
            pickle.dump(dict_No, fn)

        with io.open('/Users/ZW/Dropbox/current/temp/movie_seq_len.pickle', 'wb') as fg:
            pickle.dump(seq, fg)

        with io.open('/Users/ZW/Dropbox/current/temp/movie_sent_No.pickle', 'wb') as fsn:
            pickle.dump(sent_No, fsn)