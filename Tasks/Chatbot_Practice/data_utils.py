import nltk
from tqdm import tqdm


class DataUtils(object):
    def __init__(self):
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of Sequence
        self.eosToken = -1  # End of Sequence
        self.unknownToken = -1  # Word Dropped from Vocabulary

        self.trainingSamples = []  # 2d array containing each question and its answer [[input, target]]

    def extractConversation(self, conversations):
        for conversation in tqdm(conversations, desc='Extracing conversations'):
            