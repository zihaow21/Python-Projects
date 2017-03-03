import gensim
import pickle


# GOOGLENEWSCORPUS = "/Users/ZW/Downloads/GoogleNews-vectors-negative300.bin"
# model = gensim.models.Word2Vec.load_word2vec_format(GOOGLENEWSCORPUS, binary=True)

WIKICORPUS = "/Users/ZW/Dropbox/data/wikipedia.txt"
sentences = gensim.models.word2vec.LineSentence(WIKICORPUS, limit=1000)
# # for sentence in sentences:
# #     print sentence
model = gensim.models.Word2Vec(sentences, size=50, window=5, workers=2)
model.save("wiki_model")
#
CONVERSATIONCORPUS = "/Users/ZW/Dropbox/Current/temp/chitchat_sentence_data.txt"
new_model = gensim.models.Word2Vec.load("wiki_model")
# for key in new_model.vocab.keys():
#     print key

with open(CONVERSATIONCORPUS, 'r') as f:
    data = pickle.load(f)

model.build_vocab(data, update=True)
model.train(data)
#
print(model.similarity('cat', 'dog'))
print(model.similarity('cat', 'computer'))
model.save("wiki_model")