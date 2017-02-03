from __future__ import division

import io
import csv
from collections import OrderedDict
import random
import ast
from tqdm import tqdm

main_dir = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework0/"

file_path = main_dir + "w2_.txt"
processed_data = main_dir + "bi-gram-model.txt"
frequency = []
leading = []
following = []
bigram_prob = {}
fields = ["counts", "leading", "following"]
threshold = 0.0005

def readData(file):
    b_dict = {}
    u_dict = {}
    with io.open(file, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if line == "\n":
                continue
            else:
                values = line.strip("\n").split()
                frequency = int(values[0])
                leading = values[1]
                following = values[2]
                b_dict[(leading, following)] = frequency
                if leading not in u_dict:
                    u_dict[leading] = 0
                    u_dict[leading] += frequency
                else:
                    u_dict[leading] += frequency

    return b_dict, u_dict


def stats(file):
    b_dict, u_dict = readData(file)
    bigram_prob = {}

    for b_key in tqdm(b_dict):
        for u_key in u_dict:
            if u_key == b_key[0]:
                bigram_prob[b_key] = b_dict[b_key] / u_dict[u_key]
                if bigram_prob[b_key] > 1.0:
                    print "error"
                    break

    bigram_prob = OrderedDict(sorted(bigram_prob.items(), key=lambda d: d[0]))

    with open(processed_data, "wb") as fo:
        writer = csv.writer(fo)
        for key, value in bigram_prob.items():
            writer.writerow([key, value])

def generateSentence(processed_data):
    with open(processed_data, "rb") as fr:
        reader = csv.reader(fr)
        bigram_prob = dict(reader)

    for i in tqdm(range(10)):
        head_word = "dear"
        sentence = []
        keys = bigram_prob.keys()
        random.shuffle(keys)
        for k in range(20):
            for j, key in enumerate(keys):
                prob = float(bigram_prob[key])
                key_temp = ast.literal_eval(key)
                hw = str(key_temp[0].encode('ascii', 'ignore').decode('ascii'))
                if hw == head_word and prob >= threshold:
                    sentence.append(head_word)
                    head_word = str(key_temp[1].encode("ascii", "ignore").decode("ascii"))
                    break
        print sentence

stats(file_path)
generateSentence(processed_data)
readData(file_path)
print "job completed"
