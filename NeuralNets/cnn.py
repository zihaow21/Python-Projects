import tensorflow as tf
import numpy as np
from tqdm import tqdm


class CNN(object):
    def __init__(self, epochs, num_examples, input_dim, num_input_channels, filter_depth, filter_width, conv_stride_depth,
                 conv_stride_width, pool_stride_depth, pool_stride_width, k_depth, k_width, flat_size, num_neurons,
                 learning_rate, batch_size, num_classes, model_dir, meta_dir):
        self.epochs = epochs
        self.num_examples = num_examples
        self.input_dim = input_dim
        self.num_input_channels = num_input_channels
        self.filter_depth = filter_depth
        self.filter_width = filter_width
        self.conv_stride_depth = conv_stride_depth
        self.conv_stride_width = conv_stride_width
        self.pool_stride_depth = pool_stride_depth
        self.pool_stride_width = pool_stride_width
        self.k_depth = k_depth
        self.k_width = k_width
        self.flat_size = flat_size
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.meta_dir = meta_dir

    with tf.device(':/cpu:0'):
        def model(self, x):
            with tf.name_scope('conv1'):
                num_channels = tf.constant(1, 'int', name='num_channels')
                num_filters = tf.constant(16, 'int', name='num_filters')

                weights = tf.truncated_normal([self.conv_stride_depth, self.conv_stride_width, num_channels, num_filters], stddev=0.1, name='weights')
                biases = tf.zeros([num_filters], name='biases')
                conv = tf.nn.conv2d(x, weights, strides=[1, self.filter_depth, self.filter_width, 1], padding='VALID', name='conv')
                relu = tf.nn.relu(conv + biases, name='relu')

            with tf.name_scope('pool1'):
                pool_max = tf.nn.max_pool(relu, [1, self.k_depth, self.k_width, 1], [1, self.pool_stride_depth, self.pool_stride_width, 1], name='pool_max')

            with tf.name_scope('conv2'):
                num_channels = tf.constant(16, 'int', name='num_channels')
                num_filters = tf.constant(32, 'int', name='num_filters')

                weights = tf.truncated_normal([self.conv_stride_depth, self.conv_stride_width, num_channels, num_filters], stddev=0.1, name='weights')
                biases = tf.truncated_normal([num_filters], name='biases')
                conv = tf.nn.conv2d(pool_max, weights, strides=[1, self.filter_depth, self.filter_width, 1], padding='VALID', name='conv')
                relu = tf.nn.relu(conv + biases, name='relu')

            with tf.name_scope('pool2'):
                pool_max = tf.nn.max_pool(relu, [1, self.k_depth, self.k_width, 1], [1, self.pool_stride_depth, self.pool_stride_width, 1], name='pool_max')

            with tf.name_scope('full_connected'):
                pool_flat = np.reshape(pool_max, [-1, self.flat_size], 'pool_flat')

                weights = tf.truncated_normal([self.flat_size, self.num_neurons], stddev=0.1, name='weights')
                biases = tf.zeros([self.num_neurons], name='biases')
                relu = tf.nn.relu(tf.matmul(pool_flat, weights) + biases, name='relu')

            with tf.name_scope('output'):
                weights = tf.truncated_normal([self.num_neurons, self.num_classes], stddev=0.1, name='weights')
                biases = tf.zeros([self.num_classes], name='weights')
                output = tf.nn.relu(tf.matmul(relu, weights)+ biases, name='relu')

                return output

        def classify_train(self, data, label):
            x = tf.placeholder('float', [-1] + self.input_dim + self.num_input_channels, name='input')
            y = tf.placeholder('float', [-1, self.num_classes], name='output')
            dropout = tf.placeholder('float')

            output = self.model(x)
            correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
            cost = tf.nn.softmax_cross_entropy_with_logits(output, y)
            optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)

            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())

                for epoch in tqdm(range(self.epochs)):
                    index_shuffle = np.random.shuffle(data)
                    index_batch = index_shuffle[0: self.batch_size - 1]
                    data_batch = data[index_batch, :, :, :]
                    label_batch = label[index_batch, :]
                    if epoch % 100 == 1:
                        saver.save(sess, self.file_address)
                        sess.run(accuracy, feed_dict={x: data, y: label, dropout: 1.0})
                        print 'The training accuracy for epoch {} is {}'.format(epoch, accuracy)

                    sess.run(optimizer, feed_dict={x: data_batch, y: label_batch, dropout: 0.5})

        def classify_test(self, data, label):
            x = tf.placeholder('float', [-1] + self.input_dim + self.num_input_channels, name='input')
            y = tf.placeholder('float', [-1, self.num_classes], name='output')
            dropout = tf.placeholder('float', name='dropout')

            output = self.model(x)
            correction_prediction = tf.equal(tf.argmax(output), tf.argmax(y))
            accuracy = tf.reduce_mean(tf.cast(correction_prediction, 'float'))

            new_saver = tf.train.import_meta_graph(self.meta_dir)

            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                new_saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
                sess.run(accuracy, feed_dict={x: data, y: label, dropout: 1.0})
                print 'The testing accuracy is {}'.format(accuracy)