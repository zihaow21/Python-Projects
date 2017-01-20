###The code is modified based on https://github.com/Conchylicultor/DeepQA
import numpy as np
import nltk # for tokenize
from tqdm import tqdm
import math
import os
import random


class Vocabulary(object):
    def __init__(self):
        self.padToken = -1 # Padding
        self.goToken = -1 # Start of Sequence
        self.eosToken = -1 # End of Sequence
        self.unknownToken = -1 # Word dropped from vocabulary

        self.trainingSamples = [] # 2d array containing each question and his answer [[input, target]]

    def extractText(self, line, isTarget=False):
        """
        Extract the words from a sample line
        :param line: a lines containing text
        :param isTarget: define the answer to the question
        :return: the list of the word ids of the sentence
        """
        words = []

        sentenceToken = nltk.sent_tokenize(line)

        for i in range(len(sentenceToken)):
            if not isTarget:



    def extractConversation(self, conversation):
        for i in range(len(conversation['lines']) - 1): # ignore the last line(no answer for it
            inputLine = conversation["lines"][i]
            targetLine = conversation["lines"][i+1]

            inputWords = self.extractText(inputLine["text"])
            targetWords = self.extractText(targetLine["text"], True)

            self.trainingSamples.append([inputWords, targetWords])

    def shuffle(self):
        random.shuffle(self.trainingSamples)

