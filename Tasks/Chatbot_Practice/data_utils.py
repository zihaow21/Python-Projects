import nltk
from tqdm import tqdm


class DataUtils(object):
    def __init__(self, conversations, maxLength):
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of Sequence
        self.eosToken = -1  # End of Sequence
        self.unknownToken = -1  # Word Dropped from Vocabulary
        self.maxLength = maxLength

        self.conversations = conversations

        self.trainingSamples = []  # 2d array containing each question and its answer [[input, target], ...]

    def createCorpus(self):
        for conversation in tqdm(self.conversations, desc='Extracing conversations'):
            self.extractConversation(conversation)

    def extractConversation(self, conversation):
        for i in tqdm(range(len(conversation['lines']) - 1)):  # ignore the last line, no answer for it
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
            if len(words) + len(tokens) <= self.args.maxLength:
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

    def