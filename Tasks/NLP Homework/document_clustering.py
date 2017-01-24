from __future__ import division
import sys
from collections import Counter
import math


INPUT = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework1/docs.trn.tsv"
f = open(INPUT)

def cosine(d1, d2):
    num = den1 = den2 = 0.0

    for k, v1 in d1.items():
        den1 += v1**2
        if k in d2: num += v1 * d2[k]

    for v2 in d2.values():
        den2 += v2**2

    return float(num) / (math.sqrt(den1) * math.sqrt(den2))

tf = []

for line in f:
    t = line.split('\t')
    label = t[0]
    document = t[1].split()
    tf_doc = Counter(document)
    tf.append(tf_doc)

D = len(tf)

temp = []
for item in tf:
    temp.extend(item.keys())

df = Counter(temp)

tf_idf = []
for doc in tf:
    tf_idf_temp = {}
    for key in doc.keys():
        tf_idf_temp[key] = doc[key] * math.log(float(D)/float(df[key]))
    tf_idf.append(tf_idf_temp)

# tf is the bag of words representation for each doc, tf_idf is the tfidf representation for each doc
similarity = {}
for item1 in tf:
    for item2 in tf:
        d = cosine(item1, item2)
        if set(item1, item2) not 

print df