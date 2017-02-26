import numpy as np
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

x_text = ['This is a cat','This must be boy', 'This is a a dog']
max_document_length = max([len(x.split(" ")) for x in x_text])

## Create the vocabularyprocessor object, setting the max lengh of the documents.
vocab_processor = VocabularyProcessor(max_document_length)

## Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(x_text)))
print x

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping
print vocab_dict
## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary = list(list(zip(*sorted_vocab))[0])
print("Vocabulary : ")
print(vocabulary)
print("Transformed documents : ")
print(x)
