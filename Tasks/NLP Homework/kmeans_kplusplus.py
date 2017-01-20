import random
from numpy import linalg as nl
import numpy as np


class K_Kplusplus_Means(object):
    def __init__(self, vectors, num_random_vec):
        self.vectors = vectors
        self.length = len(self.vectors)
        self.num_random_vec = num_random_vec
        self.labels_temp = []
        self.random_int = [random.randint(0, self.length) for i in range(self.num_random_vec)]
        self.vecs_init = [vc[1] for vc in self.vectors[self.random_int]]
        self.labels = [l[0] for l in self.vectors]

    def mat_simi(self):
        norms = np.array([nl.norm(v[1]) for v in self.vectors])
        doc_matrix = np.array([t[1] for t in self.vectors])
        simi_matrix = np.array([])
        for vector in self.vecs_init:
            similarity = np.divide(np.dot(doc_matrix, np.array(vector)), norms)
            simi_matrix = np.vstack((simi_matrix, similarity))

        return simi_matrix

    def kmeans(self):
        simi_matrix = self.mat_simi()
        self.labels_temp = np.argmin(simi_matrix, axis=0)
        self.vectors = [(self.labels[i], self.labels_temp[i], vc[1] for i, vc in enumerate(self.vectors))]

        while self.labels_temp != _labels:
            self.kmeans()
            self.labels_temp = _labels



    def kmeansPlusplus(self):
        pass