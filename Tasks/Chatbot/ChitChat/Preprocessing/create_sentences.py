from cornell_data import CornellData
import nltk
import pickle


movie_lines_filename = '/Users/ZW/Dropbox/data/cornell movie-dialogs corpus/movie_lines.txt'
# movie_lines_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_lines.txt'
# movie_lines_filename = '/home/zihao/temp_folder/movie_lines.txt'

movie_conversations_filename = '/Users/ZW/Dropbox/data/cornell movie-dialogs corpus/movie_conversations.txt'
# movie_conversations_filename = '/home/zwan438/Downloads/Chitchat Data/cornell movie-dialogs corpus/movie_conversations.txt'
# movie_conversations_filename = '/home/zihao/temp_folder/movie_conversations.txt'

data_sentence_dir = '/Users/ZW/Dropbox/Current/temp/chitchat_sentence_data.txt'
# data_sentence_dir = '/home/zwan438/temp_folder/chitchat_sentence_data.txt'
# data_sentence_dir = '/home/zihao/temp_folder/chitchat_sentence_data.txt'

cd = CornellData(movie_lines_filename, movie_conversations_filename)
conversations = cd.getConversations()


class DataUtils(object):
    def __init__(self, conversations, data_sentence_dir):

        self.conversations = conversations
        self.data_sentence_dir = data_sentence_dir

        self.trainingSamples = []  # 2d array containing each question and its answer [[input, target], ...]

    def createSentenceCorpus(self):

        for conversation in self.conversations:
            self.extractSentences(conversation)

    def extractSentences(self, conversation):
        for i in range(len(conversation['lines']) - 1):  # ignore the last line, no answer for it
            inputLine = conversation['lines'][i]
            targetLine = conversation['lines'][i + 1]

            inputWords = self.sent_tokenize(inputLine['text'])
            targetWords = self.sent_tokenize(targetLine['text'])
            for sent in inputWords:
                self.trainingSamples.append(sent)
            for sent in targetWords:
                self.trainingSamples.append(sent)

    def getSampleSize(self):
        return len(self.trainingSamples)

    def sent_tokenize(self, line):

        sentences = []
        sentencesToken = nltk.sent_tokenize(line)
        for sent in sentencesToken:
            tokens = nltk.word_tokenize(sent)
            tokens = [token.lower() for token in tokens]
            sentences.append(tokens)

        return sentences

    def saveData(self):
        with open(self.data_sentence_dir, 'w') as f:
            data = self.trainingSamples
            pickle.dump(data, f)

    def loadData(self):
        with open(self.data_sentence_dir, 'r') as f:
            data = pickle.load(f)

        return data

du = DataUtils(conversations, data_sentence_dir)
du.createSentenceCorpus()
du.saveData()

