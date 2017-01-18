import io


class CornellData(object):
    def __init__(self, movie_line_fileName, movie_conversation_fileName):
        self.movie_line_fileName = movie_line_fileName
        self.movie_conversation_fileName = movie_conversation_fileName
        self.lines = {}
        self.conversations = []
        self.movie_line_fields = ["lineID", "characterID", "movieID", "character", "text"]
        self.movie_conversation_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        self.lines = self.loadLines()
        self.conversations = self.loadConversations()

    def loadLines(self):
        lines = {}

        with io.open(self.movie_line_fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split("+++$+++")

                lineObj = {}
                for i, field in enumerate(self.movie_line_fields):
                    lineObj[field] = values[i]

                lines[lineObj['lineID']] = lineObj

        return lines

    def loadConversations(self):
        conversations = []

        with io.open(self.movie_conversation_fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split("+++$+++")

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