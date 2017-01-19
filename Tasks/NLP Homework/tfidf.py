from collections import Counter
from collections import OrderedDict
import math
import numpy as np
import pickle
from tqdm import tqdm


class TFIDF(object):
    def __init__(self, main_dir, file_name, doc_vec_save):
        self.file_path = main_dir + file_name
        self.doc_vec_save = main_dir + doc_vec_save
        self.dfw_f, self.tfidf = self.fileTfIdf()

    def fileTfIdf(self):
        f = open(self.file_path)
        tf_vector = []
        dfw = Counter()
        for line in f:
            values = line.split("\t")
            label = values[0]
            document = values[1].split()
            term_frequency = (label, Counter(document))
            tf_vector.append(term_frequency)

            keys = term_frequency[1].keys()
            dfw.update(keys)

        D = float(len(tf_vector))
        dfw_f = {token: math.log(D/dfw[token]) for token in dfw.keys()}

        tfidf = []
        for label, bow in tf_vector:
            tfidf_doc = {token: count * dfw_f[token] for token, count in bow.items()}
            tfidf.append((label, tfidf_doc))

        return dfw_f, tfidf

    def vectorization(self):
        doc_vectors = []
        dfw_ordered = OrderedDict(sorted(self.dfw_f.items(), key=lambda d: d[0]))
        length = len(dfw_ordered)
        for label, tfidf_bow in tqdm(self.tfidf):
            vector = np.zeros([length, ])
            for i, key in enumerate(dfw_ordered.keys()):
                if key in tfidf_bow.keys():
                    vector[i] = tfidf_bow[key]
            doc_vectors.append((label, vector))

        f = open(self.doc_vec_save, "wb")
        pickle.dump(doc_vectors, f)
        f.close()
