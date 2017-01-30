import networkx as nx

file = "/Users/ZW/Dropbox/current/Courses/2016 NLP Homework/Homework2/word_vectors.txt"

with open(file, 'r') as f:
    data = {}
    for line in f:
        t = line.split('\t')
        data[t[0]] = t[1:]

