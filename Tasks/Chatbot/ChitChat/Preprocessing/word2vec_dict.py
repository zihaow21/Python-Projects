import gensim
import pickle


model_dir = "/Users/ZW/Dropbox/current/Python-Projects/Tasks/Chatbot_Practice/Emerson/wiki_model"
word2vec_index_dir = '/Users/ZW/Dropbox/Current/temp/word2vec_index.txt'

model = gensim.models.Word2Vec.load(model_dir)

word2vec_index_dictionary = dict()
vocab_dict = model.vocab

for key in vocab_dict.keys():
    word2vec_index_dictionary[key] = (model[key], vocab_dict[key].index)

with open(word2vec_index_dir, 'w') as f:
    pickle.dump(word2vec_index_dictionary, f)
