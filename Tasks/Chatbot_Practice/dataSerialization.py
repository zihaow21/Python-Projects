###The code is modified based on https://github.com/Conchylicultor/DeepQA
import numpy as np
import nltk # for tokenize
from tqdm import tqdm
import math
import os
import random


class Batch(object):
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

class DataSerialization(object):
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

    def sent2vec(self, conversations):
        """
        Extract data from the given vocabulary
        """
        # add standard tokens
        self.padToken = self.getWordId("<pad>")
        self.goToken = self.getWordId("<go>")
        self.eosToken = self.getWordId("<eos")
        self.unknownToken = self.getWordId("<unknown>")

        for conversation in tqdm(conversations, desc="Extracting conversations"):
            self.extractConversation(conversation)

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

    def getBatches(self):
        """
        Prepare the batches for the current epoch
        :return:
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i: min(i + self.args.batchSize, self.getSampleSize())]

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)

        return batches

    def shuffle(self):
        random.shuffle(self.trainingSamples)

    def _createBatch(self, samples):
        """
        Create a single batch from the list of samples. The batch size is automatically defined by the number of samples
        given.
        The inputs should already be inverted. The target should already have <go> and <eos>.
        :param samples: a list of samples, each sample being on the form [input, target]
        :return: Batch: a batch object
        """
        batch = Batch()
        batchSize = len(samples)

        # create the batch tensor
        for i in range(batchSize):
            sample = samples[i]
            # if not self.args.test and self.args.watsonMode: # watson mode: invert question and answer to be [target, input]
            #     sample = list(reversed(sample))
            batch.encoderSeqs.append(sample[0])
            # batch.encoderSeqs.append(list(reversed(sample[0]))) # reverse inputs (and not outputs) little trick as defined in the original paper
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])
            batch.targetSeqs.append(batch.decoderSeqs[batch.decoderSeqs[-1][1:]]) # same as decoder, but shifted to the left (ignore the <go>)

            # # Long sentences should be filtered during the dataset creation
            # assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            # assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

            # add padding and define weights
            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.padToken] * (self.args.maxLengthEnco - len(batch.encoderSeqs[i])) # right padding for the input
            # batch.encoderSeqs[i] = [self.padToken] * (self.args.maxLengthEnco - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i] # left padding for the input if input is reversed
            batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthEnco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs + [self.padToken] * (self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        # reshape the batch so that the batch of words are organized to comply with the cells in order
        encoderSeqsT = []
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []

            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        return

    def vec2sent(self, sequence, clean=False, reverse=False):
        """
        convert a list of integers into a human readable string
        :param sequence: the list of integers representing a sentence
        :param clean: indicate if removing the <go>, <pad>, and <eos> tokens
        :param reverse: for the input, option to restore the standard order
        :return: the actural sentence
        """
        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # end of the generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:
            sentence.reverse()

        return ' '.join(sentence)

    def batchVec2Sent(self, batchVec, seqId=0, **kwargs):
        """
        convert a list of integers into an actural readable sentence for a batch, which is a reshaped into a batch format
        :param batchVec: the batch format sentence
        :param seqId: the position of the sequence in a batch
        :param kwargs: the formatting options as in vec2sent
        :return: the actural sentence
        """
        sequence = []
        for i in range(len(batchVec)):
            sequence.append(batchVec[i][seqId])

        return self.vec2sent(sequence, **kwargs)

    def sent2enco(self, sentence):
        """
        encode a sequence and return a batch as an input for the model
        :return: a batch object containing the sentence, or none if something went wrong
        """
        if sentence == '':
            return None

        # devide the sentence into tokens
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # convert tokens into word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # create the vocabulary and the training sentences

        # creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # mono batch

        return batch

    def deco2sentence(self, decoderOutputs):
        """
        decode the output of the decoder and return a human friendly sentence
        :param decoderOutputs: selected words by the highest prediction score
        :return: the raw generated sentence
        """
        sequence = []

        for output in decoderOutputs:
            sequence.append(np.argmax(output))  # adding each predicted word id

        return sequence
