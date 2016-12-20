import tensorflow as tf
from tqdm import tqdm
import numpy as np
from residual_full_layer import ResLayer


class FNNRes(object):
    def __init__(self, epochs, num_layers, num_samples, input_dim, h1_nodes, h2_nodes, h3_nodes, num_classes,
                 learning_rate, batch_size, model_dir, meta_dir):
        self.epochs = epochs
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.h1_nodes = h1_nodes
        self.h2_nodes = h2_nodes
        self.h3_nodes = h3_nodes
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_lambda = 5e-4
        self.model_dir = model_dir
        self.meta_dir = meta_dir

        self.epsilon = 1e-4

    def model(self, x, dropout):

        with tf.device('/cpu:0'):
            l = ResLayer()
            prev_y = None
            y = None

            loss_l2 = tf.zeros([1], 'float')
            dim = [self.input_dim, self.h1_nodes, self.h2_nodes, self.h3_nodes, self.num_classes]

            for i in range(self.num_layers):

                if i == 0:
                    prev_y, loss = l.layer(x, dim[i], dim[i + 1], i)

                if i != 0 and i != self.num_layers - 1:
                    y_dropout = l.drop(prev_y, dropout, i)
                    prev_y = l.batch_norm(y_dropout, dim[i], i)
                    prev_y, loss = l.layer(prev_y, dim[i], dim[i + 1], i)

                if i == self.num_layers - 1:
                    y, loss = l.resFunc(prev_y, dim[i], dim[i + 1], x, self.input_dim, i)

                loss_l2 += loss

            return y, loss_l2

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

            loss = np.zeros([1, 20])

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
                    loss[0, (i-1)/5] = total_loss.eval(feed_dict={x: data_train, y: label_train, dropout: 1.0})

            return acc_train, acc_test, loss

    def regression(self, x, dropout):

        with tf.device('/cpu:0'):
            l = ResLayer()
            prev_y = None
            sig_prob = None

            loss_l2 = tf.zeros([1], 'float')
            dim = [self.input_dim, self.h1_nodes, self.h2_nodes, self.h3_nodes, self.num_classes]

            for i in range(self.num_layers):

                if i == 0:
                    prev_y, loss = l.layer(x, dim[i], dim[i + 1], i)

                if i != 0 and i != self.num_layers - 1:
                    y_dropout = l.drop(prev_y, dropout, i)
                    prev_y = l.batch_norm(y_dropout, dim[i], i)
                    prev_y = l.layer(prev_y, dim[i], dim[i + 1], i)

                if i == self.num_layers - 1:
                    sig_prob, loss = l.sigmoid(prev_y, dim[i], dim[i + 1], i)

                loss_l2 += loss

            return sig_prob, loss_l2

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

        sigmoid, loss = self.regression(x, dropout)

        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_dir)
            sess.run(tf.initialize_all_variables())
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
            prob = sigmoid.eval(feed_dict={x: data, dropout: 1.0})

            return prob