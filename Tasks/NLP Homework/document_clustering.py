from __future__ import division
import random
from collections import Counter
import numpy as np
import math


# INPUT = "/home/zwan438/temp_folder/HomeworkFiles/docs.trn.tsv"
INPUT = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework1/docs.trn.tsv"
f = open(INPUT)
kc = 7

tf = []

for i, line in enumerate(f):
    t = line.split('\t')
    label = t[0]
    document = t[1].split()
    tf_doc = (label, Counter(document))
    tf.append(tf_doc)  # in the form of [("AA", {"is": 7, "the": 6, ...}), ...]

D = len(tf)

temp = []
for ele in tf:
    temp.extend(ele[1].keys())

df = Counter(temp)

tf_idf = []  # in the form of [('AA', {'shop': 17.19, 'and': 0.38, ...}, ...]
for doc in tf:
    tf_idf_temp = {}
    for key in doc[1].keys():
        tf_idf_temp[key] = doc[1][key] * math.log(float(D)/float(df[key]))
    tf_idf.append((doc[0], tf_idf_temp))

# tf is the bag of words representation for each doc, tf_idf is the tfidf representation for each doc


def cosine(d1, d2):
    num = den1 = den2 = 0.0

    for k, v1 in d1.items():
        den1 += v1**2
        if k in d2: num += v1 * d2[k]

    for v2 in d2.values():
        den2 += v2**2

    return float(num) / (math.sqrt(den1) * math.sqrt(den2))


def clustering(item, centroids):
    dict = item[1]
    dist = []
    for centroid in centroids:
        c_dict = centroid
        dist.append(cosine(dict, c_dict))
    index = np.argmax(dist)

    return index


def mean_cal(group):
    den = len(group)
    centroid = {}
    for item in group:
        for key in item[1].keys():
            if key not in centroid:
                centroid[key] = 0
                centroid[key] += item[1][key]
            else:
                centroid[key] += item[1][key]

    for key2 in centroid.keys():
        centroid[key2] /= den

    return centroid

def compare(dist):
    distance = []
    for y in range(len(dist[0])):
        distance.append(min(np.array(dist)[:, y]))

    return distance

def centroid_add(centroids, tf_idf):
    length = len(centroids)
    dist = [[] for _ in range(length)]
    for q, centroid in enumerate(centroids):
        for it in tf_idf:
            dist[q].append(cosine(it[1], centroid))

    distance = compare(dist)
    d_2 = [distance[l] ** 2 for l in range(len(distance))]
    sum_d_2 = sum(d_2)
    d_2 = [ele / sum_d_2 for ele in d_2]
    cum = 0
    cummulative = []
    for x in range(len(d_2)):
        cum += d_2[x]
        cummulative.append(cum)

    rn = random.random()
    for v in range(len(cummulative) - 1):
        if cummulative[v] <= rn <= cummulative[v + 1]:
            return tf_idf[v + 1][1]


def centroid_init(tf_idf_ci):
    random_int_ci = random.randint(0, D)
    centroids_ci = [tf_idf_ci[random_int_ci][1]]  # in the form of {'shop': 17.19, 'and': 0.38, ...})
    for _ in range(6):
            centroids_ci.append(centroid_add(centroids_ci, tf_idf_ci))

    return centroids_ci


# kmeans ++
centroids = centroid_init(tf_idf)


# # kmeans
# random_int = [random.randint(0, D) for _ in range(kc)]
# centroids = [tf_idf[p][1] for p in random_int]  # in the form of [{'shop': 17.19, 'and': 0.38, ...}), ...]

# common
change = True
purity = 0

while change:
    cluster = [[] for _ in range(kc)]
    purity_temp = [0] * kc
    for item in tf_idf:
        cn = clustering(item, centroids)
        cluster[cn].append(item)

    for j in range(kc):
        label = []
        for item in cluster[j]:
            label.append(item[0])

        label_common = max(set(label), key=label.count)
        counter = Counter(label)
        purity_temp[j] = counter[label_common]
    print D
    print purity_temp
    p_temp = sum(purity_temp) / D

    if p_temp == purity:
        change = False

    else:
        print(purity)
        purity = p_temp

        for k in range(kc):
            centroids[k] = mean_cal(cluster[k])


