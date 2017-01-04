import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tqdm import tqdm
import numpy as np


class RNN_MultiModel_Dynamic(object):
    def __init__(self, epochs, num_samples, learning_rate, batch_size, num_classes, num_steps, vocab_size, embedding_size,
                 state_size, num_layers, model_dir, meta_dir, type):
        self.epochs = epochs
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.model_dir = model_dir
        self.meta_dir = meta_dir
        self.type = type

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                      'float', name='embeddings')

    def model(self, x, input_dropout, output_dropout):
        with tf.device('/cpu:0'):
            if self.type == 'LSTM':
                lstm_cell = rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
                dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
                multi_cell = tf.nn.rnn_cell.MultiRNNCell([dropout_cell] * self.num_layers, state_is_tuple=True)
                outputs, states = tf.nn.dynamic_rnn(multi_cell, x, dtype=tf.float32)
                outputs = [tf.squeeze(item, [0]) for item in tf.split(0, self.num_steps, outputs)]

            if self.type == 'GRU':
                gru_cell = rnn_cell.GRUCell(self.state_size)
                dropout_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
                multi_cell = tf.nn.rnn_cell.MultiRNNCell([dropout_cell] * self.num_layers, state_is_tuple=True)
                outputs, states = tf.nn.dynamic_rnn(multi_cell, x, dtype=tf.float32)
                outputs = [tf.squeeze(item, [0]) for item in tf.split(0, self.num_steps, outputs)]

            return outputs

    def classify_train_test(self, data_train, label_train, data_test, label_test):
        x = tf.placeholder(tf.int32, [None, self.num_steps], name='input_placeholder')
        y = tf.placeholder('float', [None, self.num_classes], name='labels_placeholder')
        input_dropout = tf.placeholder('float')
        output_dropout = tf.placeholder('float')

        inputs = tf.nn.embedding_lookup(self.embeddings, x)
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = [tf.squeeze(item, [0]) for item in tf.split(0, self.num_steps, inputs)]

        with tf.name_scope('softmax'):
            W = tf.Variable(tf.truncated_normal([self.state_size, self.num_classes], stddev=0.1), 'float', name='W')
            b = tf.Variable(tf.zeros([self.num_classes]), name='b')
            outputs = self.model(inputs, input_dropout, output_dropout)
            output_last = outputs[- 1]
            logits = tf.matmul(output_last, W) + b
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

            correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                saver = tf.train.Saver()

                for i in tqdm(range(self.epochs)):
                    index_shuffle = np.random.permutation(self.num_samples)
                    index_batch = index_shuffle[0: self.batch_size - 1]
                    batch_data = data_train[index_batch, :]
                    batch_label = label_train[index_batch, :]
                    sess.run(optimizer, feed_dict={x: batch_data, y: batch_label, input_dropout: 1.0, output_dropout: 0.5})

                    if i % 5 == 1:
                        acc_n = accuracy.eval(feed_dict={x: batch_data, y: batch_label, input_dropout: 1.0, output_dropout: 1.0})
                        print '\n The training accuracy for epoch {} is {}'.format(i, acc_n)
                        acc_t = accuracy.eval(feed_dict={x: data_test, y: label_test, input_dropout: 1.0, output_dropout: 1.0})
                        print '\n The testing accuracy for epoch {} is {}'.format(i, acc_t)
                        saver.save(sess, self.model_dir)
