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