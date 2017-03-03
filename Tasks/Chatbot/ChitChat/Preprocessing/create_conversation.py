from cornell_data import CornellData
import nltk
import pickle
import numpy as np
import gensim
from tqdm import tqdm


# word2vec_index_dir = '/Users/ZW/Dropbox/Current/temp/word2vec_dict.txt'

word2vec_index_dir = '/home/zwan438/Dropbox/data/word2vec_dict.txt'

# movie_lines_filename = '/Users/ZW/Dropbox/data/cornell movie-dialogs corpus/movie_lines.txt'
movie_lines_filename = '/home/zwan438/Dropbox/data/cornell movie-dialogs corpus/movie_lines.txt'
# movie_lines_filename = '/home/zihao/temp_folder/movie_lines.txt'

# movie_conversations_filename = '/Users/ZW/Dropbox/data/cornell movie-dialogs corpus/movie_conversations.txt'
movie_conversations_filename = '/home/zwan438/Dropbox/data/cornell movie-dialogs corpus/movie_conversations.txt'
# movie_conversations_filename = '/home/zihao/temp_folder/movie_conversations.txt'

# data_conversation_dir = '/Users/ZW/Dropbox/Current/temp/chitchat_conversation_data.txt'
data_conversation_dir = '/home/zwan438/Dropbox/data/chitchat_conversation_data.txt'
# data_conversation_dir = '/home/zihao/temp_folder/chitchat_conversation_data.txt'

cd = CornellData(movie_lines_filename, movie_conversations_filename)
conversations = cd.getConversations()

with open(word2vec_index_dir, 'r') as f:
    vector_model = pickle.load(f)
print "loading data finished"
vocab_len = len(vector_model)
EMBEDDING_DIM = 50

vector_model["<go>"] = (np.random.uniform(-0.25, 0.25, EMBEDDING_DIM), vocab_len + 1)
vector_model["<eos>"] = (np.random.uniform(-0.25, 0.25, EMBEDDING_DIM), vocab_len + 2)
vector_model["<unknown>"] = (np.random.uniform(-0.25, 0.25, EMBEDDING_DIM), vocab_len)
vector_model["<pad>"] = (np.random.uniform(-0.25, 0.25, EMBEDDING_DIM), vocab_len + 3)

class DataUtils(object):
    def __init__(self, conversations, maxLength, vector_model, data_conversation_dir):
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of Sequence
        self.eosToken = -1  # End of Sequence
        self.unknownToken = -1  # Word Dropped from Vocabulary
        self.maxLength = maxLength
        self.data_conversation_dir = data_conversation_dir

        self.vector_model = vector_model

        self.word2id = {}
        self.id2word = {}

        self.conversations = conversations

        self.trainingSamples = []  # 2d array containing each question and its answer [[input, target], ...]

    def createCorpus(self):
        # add standard tokens
        self.padToken = self.getWordId("<pad>")
        self.goToken = self.getWordId("<go>")
        self.eosToken = self.getWordId("<eos>")
        self.unknownToken = self.getWordId("<unknown>")
        print "conversation length is {}".format(len(self.conversations))
        for conversation in tqdm(self.conversations):
            self.createConversation(conversation)

    def createConversation(self, conversation):
        length = len(conversation['lines']) - 1
        for i in range(length):  # ignore the last line, no answer for it
            inputLine = conversation['lines'][i]
            targetLine = conversation['lines'][i+1]

            inputWords = self.extractText(inputLine['text'])
            targetWords = self.extractText(targetLine['text'], True)

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
                i = len(sentencesToken) - 1 - i  # if question, starting to form the sentence from the last question

            tokens = nltk.word_tokenize(sentencesToken[i])

            # keep adding sentences until reaching the preset maximum length of the sentence
            if len(words) + len(tokens) <= self.maxLength:
                tempWords = []

                for token in tokens:
                    tempWords.append(self.getWordId(token))  # create the vocabulary and the training sentences

                if isTarget:
                    words = words + tempWords  # words is the previous answer + tempWords is the later answer
                else:
                    words = tempWords + words  # temWords is the previous question + words is the later question

            else:
                break
        return words

    def getWordId(self, word):
        """
        Get the id of the word (and add it to the dictionary if not  existing). If the word does not exist and create is
        set to False, the function will return the unknownToken value.
        :param word: the input word
        :param create: the word will be added to the dict
        :return: id of the word
        """
        word = word.lower()
        if word in self.vector_model.keys():
            wordId = self.vector_model[word][1]
            self.word2id[word] = wordId
            self.id2word[wordId] = word
        else:
            wordId = -1

        return wordId

    def saveData(self):
        print "start saving data"
        data = {
            'word2id': self.word2id,
            'id2word': self.id2word,
            'trainingSamples': self.trainingSamples
        }
        with open(self.data_conversation_dir, 'w') as f:
            pickle.dump(data, f)

    # def loadData(self):
    #     with open(self.data_conversation_dir, 'r') as f:
    #         data = pickle.load(f)
    #         self.word2id = data['word2id']
    #         self.id2word = data['id2word']
    #         self.trainingSamples = data['trainingSamples']

du = DataUtils(conversations, 40, vector_model, data_conversation_dir)
du.createCorpus()
print "corpus created"
print "now saving data"
du.saveData()
