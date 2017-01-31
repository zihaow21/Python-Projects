from __future__ import division
import math
import numpy as np
from tqdm import tqdm


# file = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework2/word_vectors.txt"
file = "/home/zwan438/temp_folder/HomeworkFiles/word_vectors.txt"

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

for item1 in tqdm(data):
    S[item1[0]] = []
    visited.append(item1)

    for i, item2 in enumerate(data):
        if item2 not in visited:
            S[item1[0]].append([i, cosines(item1[1], item2[1])])

    if S[item1[0]]:
        list_compare = list(np.array(S[item1[0]])[:, 1])
        id = list_compare.index(max(list_compare))
        final_list.append([item1[0], data[S[item1[0]][id][0]][0], S[item1[0]][id][1]])

with open("/home/zwan438/temp_folder/HomeworkFiles/finallist.txt", 'wb') as ffw:
    for item in final_list:
        ffw.write("{}\n".format(item))

# with open("/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework2/sample.dot", "wb") as fout:
with open("/home/zwan438/temp_folder/HomeworkFiles/sample.dot", "wb") as fout:
    fout.write('digraph G {\n')
    for item in final_list:
        fout.write('"{}" -> "{}" [label="{}"];\n'.format(item[0], item[1], 1.0 - item[2]))
    fout.write('}\n')

with open("/home/zwan438/temp_folder/HomeworkFiles/finallist.txt", 'rb') as ffr:
    for line in ffr:
        final_list.append(eval(line))

threshold = 0.88
classes = []
classes.append([])
c = 0
visited2 = []
while final_list:
    if not classes[c]:
        vector_matrix = np.array(final_list)
        id = np.argmax(list(vector_matrix[:, 2]))
        max_value = max(list(vector_matrix[:, 2]))
        if max_value >= threshold:
            classes[c].extend(final_list[id][0: 2])
            visited2.extend(final_list[id][0: 2])
            final_list.pop(id)
        else:
            break

    else:
        left = list(np.array(final_list)[:, 0])
        if classes[c][-1] in left and classes[c][-1] not in visited2:
            pair_index = list(np.array(final_list)[:, 0]).index(classes[c][-1])
            pair = final_list[pair_index]
            if pair[2] >= threshold:
                classes[c].extend(pair[0: 2])
                visited2.extend(final_list[id][0: 2])
                final_list.pop(pair_index)
        else:
            classes.append([])
            c += 1

print classes