import nltk
import random
import numpy as np
import pickle


class Batch:
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

class DataUtils(object):
    def __init__(self, word2vec_dict_dir, maxLength, maxLengthEnco, maxLengthDeco, batchSize, data_conversation_dir):
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of Sequence
        self.eosToken = -1  # End of Sequence
        self.unknownToken = -1  # Word Dropped from Vocabulary
        self.maxLength = maxLength
        self.maxLengthEnco = maxLengthEnco
        self.maxLengthDeco = maxLengthDeco
        self.batchSize = batchSize
        self.data_conversation_dir = data_conversation_dir
        self.word2vec_index_dir = word2vec_dict_dir

        with open(data_conversation_dir, 'r') as f:
            data = pickle.load(f)
            self.trainingSamples = data['trainingSamples']
            self.word2id = data['word2id']
            self.id2word = data['id2word']

    def shuffle(self):
        random.shuffle(self.trainingSamples)

    def getSampleSize(self):
        return len(self.trainingSamples)

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
            wordId = self.unknownToken

        return wordId

    def getBatches(self):
        self.shuffle()
        batches = []

        def genNextSamples():
            for i in range(0, self.getSampleSize(), self.batchSize):
                yield self.trainingSamples[i: min(i + self.batchSize, self.getSampleSize())]

        for samples in genNextSamples():
            batch = self.createBatch(samples)
            batches.append(batch)

        return batches

    def createBatch(self, samples):
        batch = Batch()

        batchSize = len(samples)

        for i in range(batchSize):
            sample = samples[i]
            batch.encoderSeqs.append(sample[0])
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])
            batch.targetSeqs.append(batch.decoderSeqs[-1][1:])

            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.padToken] * (self.maxLengthEnco - len(batch.encoderSeqs[i]))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (self.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.padToken] * (self.maxLengthDeco - len(batch.targetSeqs[i]))
            batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.maxLengthDeco - len(batch.targetSeqs[i])))

        encoderSeqsT = []
        for i in range(self.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.maxLengthDeco):
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

        return batch

    def vec2str(self, sequence):
        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        return ' '.join(sentence)

    def deco2vec(self, decoderOutputs):
        """
        decode the output of the decoder and return a human friendly sentence
        :param decoderOutputs: selected words by the highest prediction score
        :return: the raw generated sentence
        """
        sequence = []

        for output in decoderOutputs:
            sequence.append(np.argmax(output))  # adding each predicted word id

        return sequence

    def sent2enco(self, sentence):
        """
        encode a sequence and return a batch as an input for the model
        :return: a batch object containing the sentence, or none if something went wrong
        """
        if sentence == '':
            return None

        # devide the sentence into tokens
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.maxLength:
            return None

        # convert tokens into word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # create the vocabulary and the training sentences

        # creating the batch (add padding, reverse)
        batch = self.createBatch([[wordIds, []]])  # mono batch

        return batch