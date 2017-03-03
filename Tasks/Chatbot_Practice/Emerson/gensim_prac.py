import gensim
import pickle


# WIKICORPUS = "/Users/ZW/Dropbox/data/wikipedia.txt"
WIKICORPUS = "/home/zwan438/Dropbox/data/wikipedia.txt"
sentences = gensim.models.word2vec.LineSentence(WIKICORPUS, limit=1000)
# # for sentence in sentences:
# #     print sentence
model = gensim.models.Word2Vec(sentences, size=50, window=5, workers=2)
model.save("wiki_model")
#
# CONVERSATIONCORPUS = "/Users/ZW/Dropbox/Current/temp/chitchat_sentence_data.txt"
CONVERSATIONCORPUS = "/home/zwan438/Dropbox/chitchat_sentence_data.txt"
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
