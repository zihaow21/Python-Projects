from __future__ import division
import networkx as nx
import  graphviz as gz
import numpy as np
import pandas as pd
from itertools import groupby


file = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework2/word_vectors.txt"

with open(file, 'r') as f:
    data = []
    for line in f:
        t = line.split('\t')
        data.append([t[0], t[1:]])


def cosines(d1, d2):
    num = 0
    den1 = 0
    den2 = 0

    for e1, e2 in zip(d1, d2):
        num += float(e1) * float(e2)
        den1 += (float(e1) ** 2) ** 1/2
        den2 += (float(e2) ** 2) ** 1/2

    similarity = num / (den1 * den2)

    return similarity

S = []
for item1 in data:
    for item2 in data:
        S.append([item1[0], item2[0], cosines(item1[1], item2[1])])

field = ['lead', 'follow', 'distance']
for item in S:
