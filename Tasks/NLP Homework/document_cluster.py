from tfidf import TFIDF
import pickle
from kmeans_kplusplus import K_Kplusplus_Means


main_dir = "/home/zwan438/temp_folder/HomeworkFiles/"
file_name = "docs.trn.tsv"
doc_vec_save = "doc_vec_save.pkl"

# tf_idf = TFIDF(main_dir, file_name, doc_vec_save)
# tf_idf.vectorization()

doc_vec_f = open(main_dir + doc_vec_save, "rb")
doc_vectors = pickle.load(doc_vec_f)
doc_vec_f.close()

num_init_vec = 7
kpm = K_Kplusplus_Means(doc_vectors, num_init_vec)



