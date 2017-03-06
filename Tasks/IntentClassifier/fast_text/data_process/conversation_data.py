import io
import pickle


class CornellData(object):
    def __init__(self, movie_lines_fileName, movie_conversations_fileName):
        self.movie_line_fileName = movie_lines_fileName
        self.movie_conversation_fileName = movie_conversations_fileName
        self.lines = {}
        self.conversations = []
        self.movie_line_fields = ["lineID", "characterID", "movieID", "character", "text"]
        self.movie_conversation_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        self.lines = self.loadLines()
        self.conversations = self.loadConversations()
        self.data = []
        self.extractConversation()
        self.utterance = dict()
        self.conversationIndex()
        self.utterance_label = []
        self.utteranceLabel()


    def loadLines(self):
        lines = {}

        with io.open(self.movie_line_fileName, 'r', errors='ignore') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                lineObj = {}
                for i, field in enumerate(self.movie_line_fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj

        return lines

    def loadConversations(self):
        conversations = []

        with io.open(self.movie_conversation_fileName, 'r', errors='ignore') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                convObj = {}
                for i, field in enumerate(self.movie_conversation_fields):
                    convObj[field] = values[i]

                lineIDs = eval(convObj["utteranceIDs"])
                convObj["lines"] = []
                for lineID in lineIDs:
                    convObj["lines"].append(self.lines[lineID])

                conversations.append(convObj)

        return conversations

    def getConversations(self):

        return self.conversations

    def conversationIndex(self):
        c = 0
        for i, conversation in enumerate(self.conversations):
            for j, conv in enumerate(conversation['lines']):
                self.utterance[conv['text']] = c
                c += 1

    def extractConversation(self):
        for conversation in self.conversations:
            for i in range(len(conversation['lines']) - 1):  # ignore the last line, no answer for it
                inputLine = conversation['lines'][i]
                targetLine = conversation['lines'][i+1]
                self.data.append([inputLine['text'], targetLine['text']])

    def utteranceLabel(self):
        for l in self.data:
            label = self.utterance[l[1]]
            self.utterance_label.append(''.join(['__label__', str(label), ', ', l[0], '\n']).encode('utf-8').strip())

movie_lines_fileName = '/Users/ZW/Dropbox/data/cornell movie-dialogs corpus/movie_lines.txt'
movie_conversations_fileName = '/Users/ZW/Dropbox/data/cornell movie-dialogs corpus/movie_conversations.txt'
conversation_index_fileName = '/Users/ZW/Dropbox/data/conversation_index.txt'
conversation_train_fileName = '/Users/ZW/Dropbox/data/conversation_train.txt'
cd = CornellData(movie_lines_fileName, movie_conversations_fileName)

conversations = cd.getConversations()
conversation_data = cd.data
conversation_index = cd.utterance
conversation_label = cd.utterance_label
with open(conversation_index_fileName, 'wb') as f:
    pickle.dump(conversation_index, f)
with open(conversation_train_fileName, 'wb') as f:
    for l in conversation_label:
        print >> f, l
    f.close()
