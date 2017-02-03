import tensorflow as tf


class ProjectOp(object):
    def __init__(self, dim):
        with tf.name_scope('projection'):
            self.W = tf.Variable(tf.truncated_normal(dim, stddev=0.1), name='weights')
            self.b = tf.Variable(tf.zeros([dim[1]]), name='biases')

    def projection(self, x):
        with tf.name_scope('projection'):
            target_complete = tf.matmul(x, self.W) + self.b

        return target_complete

class Seq2seq(object):
    def __init__(self, epochs, learning_rate, batch_size, vocab_size, num_softmax_samples, embedding_size, model_dir, meta_dir):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_softmax_samples = num_softmax_samples
        self.embedding_size = embedding_size
        self.model_dir = model_dir
        self.meta_dir = meta_dir

    def model(self, x):
        if 0 < self.num_softmax_samples < self.vocab_size:

