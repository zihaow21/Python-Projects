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

