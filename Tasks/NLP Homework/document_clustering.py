from collections import Counter
import math
import sys

main_dir = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework1/"
file_path = main_dir + "docs.trn.tsv"


def readFile(file_path):
    f = open(file_path)
    vector_bow = list()
    dfw = Counter()

    for line in f:
        t = line.split('\t')
        label = t[0]
        document = t[1]
        vector = (label, Counter(document.split()))
        vector_bow.append(vector)
        keys = vector[1].keys()
        dfw.update(keys)

    D = float(len(vector_bow))

    for token in dfw:
        dfw[token] = math.log(D/dfw[token])

    vector_tfidf = list()

    for (label, bow) in vector_bow:
        tfidf = {token: dfw[token] * count for (token, count) in bow.items()}
        vector_tfidf.append((label, tfidf))

    return vector_tfidf

def similarity(d1, d2):
    num = den1 = den2 = 0.0

    for k, v1 in d1.items():
        den1 += v1 ** 2
        if k in d2:
            num += v1 * d2[k]

    for v2 in d2.items():
        den2 += v2 ** 2

    return float(num) / (math.sqrt(den1) * math.sqrt(den2))

def kmeans():
    pass
vector_tfidf = readFile(file_path)

