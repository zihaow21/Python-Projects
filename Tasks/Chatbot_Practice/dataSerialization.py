###The code is modified based on https://github.com/Conchylicultor/DeepQA
import numpy as np
import nltk # for tokenize
from tqdm import tqdm
import math
import os
import random


class Vocabulary(object):
    def __init__(self, args):
        self.args = args

        self.padToken = -1 # Padding
        self.goToken = -1 # Start of Sequence
        self.eosToken = -1 # End of Sequence
        self.unknownToken = -1 # Word dropped from vocabulary

        self.word2id = {}
        self.id2word = {}

        self.trainingSamples = [] # 2d array containing each question and his answer [[input, target]]

    def getSampleSize(self):

        return len(self.trainingSamples)

    def getVocabSize(self):

        return len(self.word2id)

    def extractConversation(self, conversation):
        for i in range(len(conversation['lines']) - 1): # ignore the last line(no answer for it
            inputLine = conversation["lines"][i]
            targetLine = conversation["lines"][i+1]

            inputWords = self.extractText(inputLine["text"])
            targetWords = self.extractText(targetLine["text"], True)

            self.trainingSamples.append([inputWords, targetWords])

    def extractText(self, line, isTarget=False):
        """
        Extract the words from a sample line
        :param line: a lines containing text
        :param isTarget: define the answer or the question, if true, answer, if false question
        :return: the list of the word ids of the sentence
        """
        words = []

        sentencesToken = nltk.sent_tokenize(line)

        for i in range(len(sentencesToken)):
            # if question: we only keep the last sentences up to the preset maximum length of the sentence
            # if answer: we only keep the first sentences up to the preset maximum length of the sentence

            if not isTarget:
                i = len(sentencesToken) - 1 - i # if question, starting to form the sentence from the last question

            tokens = nltk.word_tokenize(sentencesToken[i])

            # keep adding sentences until reaching the preset maximum length of the sentence
            if len(words) + len(tokens) <= self.args.maxLength:
                tempWords = []

                for token in tokens:
                    tempWords.append(self.getWordId(token)) # create the vocabulary and the training sentences

                if isTarget:
                    words = words + tempWords
                else:
                    words = tempWords + words

            else:
                break

        return words

    def getWordId(self, word, create=True):
        """
        Get the id of the word (and add it to the dictionary if not  existing). If the word does not exist and create is
        set to False, the function will return the unknownToken value.
        :param word: the input word
        :param create: the word will be added to the dict
        :return: id of the word
        """
        word = word.lower()
        wordId = self.word2id.get(word, -1)

        if wordId == -1:
            if create:
                wordId = len(self.word2id)
                self.word2id[word] = wordId
                self.id2word[wordId] = word
            else:
                wordId = self.unknownToken

        return wordId

    def


    def shuffle(self):
        random.shuffle(self.trainingSamples)

