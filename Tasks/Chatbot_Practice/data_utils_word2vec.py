import nltk
import random
import numpy as np
import pickle
import gensim


class Batch:
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

class DataUtils(object):
    def __init__(self, conversations, maxLength, maxLengthEnco, maxLengthDeco, batchSize, data_dir):
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of Sequence
        self.eosToken = -1  # End of Sequence
        self.unknownToken = -1  # Word Dropped from Vocabulary
        self.maxLength = maxLength
        self.maxLengthEnco = maxLengthEnco
        self.maxLengthDeco = maxLengthDeco
        self.batchSize = batchSize
        self.data_dir = data_dir

        self.word2id = {}
        self.id2word = {}

        self.conversations = conversations

        self.trainingSamples = []  # 2d array containing each question and its answer [[input, target], ...]

    def createSentenceCorpus(self):

        for conversation in self.conversations:
            self.extractSentences(conversation)

    def extractSentences(self, conversation):
        for i in range(len(conversation['lines']) - 1):  # ignore the last line, no answer for it
            inputLine = conversation['lines'][i]
            targetLine = conversation['lines'][i+1]

            inputWords = self.sent_tokenize(inputLine['text'])
            targetWords = self.sent_tokenize(targetLine['text'])
            for sent in inputWords:
                self.trainingSamples.append(sent)
            for sent in targetWords:
                self.trainingSamples.append(sent)

    def getSampleSize(self):
        return len(self.trainingSamples)

    def sent_tokenize(self, line):
        """
        Extract the words from a sample line
        :param line: a lines containing text
        :param isTarget: define the answer or the question, if true, answer, if false question
        :return: the list of the word ids of the sentence
        """
        sentences = []
        sentencesToken = nltk.sent_tokenize(line)
        for sent in sentencesToken:
            tokens = nltk.word_tokenize(sent)
            sentences.append(tokens)

        return sentences

    def saveData(self):
        with open(self.data_dir, 'w') as f:
            data = self.trainingSamples
            pickle.dump(data, f)

    def loadData(self):
        with open(self.data_dir, 'r') as f:
            data = pickle.load(f)

        return data