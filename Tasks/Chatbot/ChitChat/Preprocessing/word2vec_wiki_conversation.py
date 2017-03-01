import gensim
import pickle


WIKICORPUS = "/Users/ZW/Downloads/a5/wikipedia.txt"
CONVERSATIONCORPUS = "/Users/ZW/Dropbox/Current/temp/chitchat_sentence_data.txt"
with open(CONVERSATIONCORPUS, 'r') as f:
    data = pickle.load(f)

sentences = gensim.models.word2vec.LineSentence(WIKICORPUS)
corpus_complete = data
for sentence in sentences:
    corpus_complete += sentences

with open("/Users/ZW/Dropbox/Current/temp/complete_corpus.txt", "w") as f:
    pickle.dump(corpus_complete, f)

with open("/Users/ZW/Dropbox/Current/temp/complete_corpus.txt", "r") as fr:
    corpus = pickle.load(fr)

model = gensim.models.Word2Vec(corpus, size=50, window=5, workers=2)

model.save("wiki_conversation_model")

model = gensim.models.Word2Vec.load("wiki_conversation_model")
print(model.similarity('cat', 'dog'))
print(model.similarity('cat', 'computer'))