from __future__ import division
import math
import numpy as np


file = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework2/word_vectors.txt"
# file = "/home/zwan438/temp_folder/HomeworkFiles/word_vectors.txt"

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
        den1 += float(e1) ** 2
        den2 += float(e2) ** 2

    similarity = num / (math.sqrt(den1) * math.sqrt(den2))

    return similarity

S = {}
visited = []
final_list = []
for item1 in data:
    S[item1[0]] = []
    visited.append(item1)
    for i, item2 in enumerate(data):
        if item2 not in visited:
            S[item1[0]].append([i, cosines(item1[1], item2[1])])
    if S[item1[0]] != []:
        list_compare = list(np.array(S[item1[0]])[:, 1])
        id = list_compare.index(max(list_compare))
        final_list.append([item1[0], data[S[item1[0]][id][0]][0], S[item1[0]][id][1]])

with open("/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework2/sample.dot", "wb") as fout:
# with open("/home/zwan438/temp_folder/HomeworkFiles/sample.dot", "wb") as fout:
    fout.write('digraph G {\n')
    for item in final_list:
        fout.write('"{}" -> "{}" [label="{}"];\n'.format(item[0], item[1], item[2]))
    fout.write('}\n')

