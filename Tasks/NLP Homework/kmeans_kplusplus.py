import random
from numpy import linalg as nl
import numpy as np


class K_Kplusplus_Means(object):
    def __init__(self, vectors, num_random_vec):
        self.vectors = vectors
        self.length = len(self.vectors)
        self.num_random_vec = num_random_vec

    def mat_simi(self):
        random_int = [random.randint(0, self.length) for i in range(self.num_random_vec)]
        vecs_init = [vc[1] for vc in self.vectors[random_int]]

        labels = [l[0] for l in self.vectors]
        norms = np.array([nl.norm(v[1]) for v in self.vectors])
        doc_matrix = np.array([t[1] for t in self.vectors])
        simi_matrix = np.array([])
        for vector in vecs_init:
            similarity = np.divide(np.dot(doc_matrix, np.array(vector)), norms)
            simi_matrix = np.vstack((simi_matrix, similarity))

        return vecs_init, simi_matrix, labels

    def kmeans(self):
        centroids, simi, labels = self.mat_simi()


    def kmeansPlusplus(self):
        pass