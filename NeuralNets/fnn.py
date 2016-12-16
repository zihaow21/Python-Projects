import tensorflow as tf
import math
from tqdm import tqdm
import numpy as np


class FNN(object):
    def __init__(self, epochs, num_samples, input_dim, h1_nodes, h2_nodes, h3_nodes, num_classes,
                 learning_rate, batch_size, l2_lambda, model_dir, meta_dir):
        self.epochs = epochs
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.h1_nodes = h1_nodes
        self.h2_nodes = h2_nodes
        self.h3_nodes = h3_nodes
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.model_dir = model_dir
        self.meta_dir = meta_dir

        self.epsilon = 1e-4

    def model(self, x, dropout):

        with tf.device('/cpu:0'):
            loss_l2 = tf.zeros([1], 'float')

            with tf.name_scope('h1'):
                weights = tf.Variable(tf.truncated_normal([self.input_dim, self.h1_nodes],
                                                stddev=1.0 / math.sqrt(float(self.input_dim))), 'float', name='weights')
                biases = tf.Variable(tf.zeros([self.h1_nodes]), 'float', name='biases')
                h1 = tf.nn.relu(tf.matmul(x, weights) + biases, name='relu')
                h1_dropout = tf.nn.dropout(h1, dropout, name='dropout')

                batch_mean, batch_var = tf.nn.moments(h1_dropout, [0])
                scale = tf.Variable(tf.ones([self.h1_nodes]), name='scale')
                shift = tf.Variable(tf.zeros([self.h1_nodes]), name='shift')
                h1_batch_norm = tf.nn.batch_normalization(h1_dropout, batch_mean, batch_var, shift,
                                                          scale, self.epsilon, name='batch_norm')

                loss_l2 += self.l2_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

            with tf.name_scope('h2'):
                weights = tf.Variable(tf.truncated_normal([self.h1_nodes, self.h2_nodes],
                                                stddev=1.0 / math.sqrt(float(self.h1_nodes))), 'float', name='weights')
                biases = tf.Variable(tf.zeros([self.h2_nodes]), 'float', name='biases')
                h2 = tf.nn.relu(tf.matmul(h1_batch_norm, weights) + biases, name='relu')
                h2_dropout = tf.nn.dropout(h2, dropout, name='dropout')

                batch_mean, batch_var = tf.nn.moments(h2_dropout, [0])
                scale = tf.Variable(tf.ones([self.h2_nodes]), name='scale')
                shift = tf.Variable(tf.zeros([self.h2_nodes]), name='shift')
                h2_batch_norm = tf.nn.batch_normalization(h2_dropout, batch_mean, batch_var, shift,
                                                          scale, self.epsilon, name='batch_norm')

                loss_l2 += self.l2_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

            with tf.name_scope('softmax'):
                weights = tf.Variable(tf.truncated_normal([self.h2_nodes, self.num_classes],
                                                stddev=1.0 / math.sqrt(float(self.h2_nodes))), 'float', name='weights')
                biases = tf.Variable(tf.zeros([self.num_classes]), 'float', name='biases')
                logits = tf.matmul(h2_batch_norm, weights) + biases

                loss_l2 = self.l2_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

            return logits, loss_l2

    def classify_train_test(self, data_train, label_train, data_test, label_test):
        x = tf.placeholder('float', [None, self.input_dim])
        y = tf.placeholder('float', [None, self.num_classes])
        dropout = tf.placeholder('float')

        logits, loss_l2 = self.model(x, dropout)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
        total_loss = loss_cross_entropy + loss_l2
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=total_loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            acc_train = np.zeros([1, 20])
            acc_test = np.zeros([1, 20])

            for i in tqdm(range(self.epochs)):
                index_shuffle = np.random.permutation(self.num_samples)
                index_batch = index_shuffle[0: self.batch_size - 1]
                batch_data = data_train[index_batch, :]
                batch_label = label_train[index_batch, :]

                sess.run(optimizer, feed_dict={x: batch_data, y: batch_label, dropout: 0.5})
                if i % 5 == 1:
                    acc_train[0, (i-1)/5] = accuracy.eval(feed_dict={x: data_train, y: label_train, dropout: 1.0})
                    print '\n The training accuracy for epoch {} is {}'.format(i, acc_train[0, (i-1)/5])
                    acc_test[0, (i-1)/5] = accuracy.eval(feed_dict={x: data_test, y: label_test, dropout: 1.0})
                    print '\n The testing accuracy for epoch {} is {}'.format(i, acc_test[0, (i-1)/5])
                    saver.save(sess, self.model_dir)

            return acc_train, acc_test

    def regression(self, x, dropout):
        loss_l2 = tf.zeros([1], 'float')
        with tf.device('/cpu:0'):
            with tf.name_scope('h1'):
                weights = tf.Variable(tf.truncated_normal([self.input_dim, self.h1_nodes],
                                    stddev=1.0 / math.sqrt(float(self.input_dim))), 'float', name='weights')
                biases = tf.Variable(tf.zeros([self.h1_nodes]), 'float', name='biases')
                h1 = tf.nn.relu(tf.matmul(x, weights) + biases)
                h1_dropout = tf.nn.dropout(h1, dropout)

                loss_l2 += self.l2_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

            with tf.name_scope('h2'):
                weights = tf.Variable(tf.truncated_normal([self.h1_nodes, self.h2_nodes],
                                    stddev=1.0 / math.sqrt(float(self.h1_nodes))), 'float', name='weights')
                biases = tf.Variable(tf.zeros([self.h2_nodes]), 'float', name='biases')
                h2 = tf.nn.relu(tf.matmul(h1_dropout, weights) + biases)
                h2_dropout = tf.nn.dropout(h2, dropout)

                loss_l2 += self.l2_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

            with tf.name_scope('h3'):
                weights = tf.Variable(tf.truncated_normal([self.h2_nodes, self.h3_nodes],
                                    stddev=1.0 / math.sqrt(float(self.h2_nodes))), 'float', name='weights')
                biases = tf.Variable(tf.zeros([self.h3_nodes]), 'float', name='biases')
                h3 = tf.nn.relu(tf.matmul(h2_dropout, weights) + biases)
                h3_dropout = tf.nn.dropout(h3, dropout)

                loss_l2 += self.l2_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

            with tf.name_scope('sigmoid'):
                weights = tf.Variable(tf.truncated_normal([self.h2_nodes, 1],
                                    stddev=1.0 / math.sqrt(float(self.h1_nodes))), 'float', name='weights')
                biases = tf.Variable(tf.zeros([1]), 'float', name='biases')
                y_ = tf.matmul(h2_dropout, weights) + biases
                sigmoid = tf.nn.sigmoid(y_)

                loss_l2 = self.l2_lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))

            return sigmoid, loss_l2

    def classify_test(self, data, label):
        x = tf.placeholder('float', [None, self.input_dim])
        y = tf.placeholder('float', [None, self.num_classes])
        dropout = tf.constant(1.0, 'float')

        logits, _ = self.model(x, dropout)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        new_saver = tf.train.import_meta_graph(self.meta_dir)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
            accuracy.eval(feed_dict={x: data, y: label, dropout: 1.0})

            print 'The testing accuracy is {}'.format(accuracy)

    def regression_test(self, data):
        x = tf.placeholder('float', [None, self.input_dim])
        dropout = tf.constant(1.0, 'float')

        sigmoid, loss = self.model(x, dropout)

        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_dir)
            sess.run(tf.initialize_all_variables())
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
            prob = sigmoid.eval(feed_dict={x: data, dropout: 1.0})

            return prob