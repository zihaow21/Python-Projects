import tensorflow as tf


class ResLayer(object):
    def __init__(self):
        self.epsilon = 1e-4
        self.l2_lambda = 5e-4

    def layer(self, x, input_dim, output_dim, depth):
        with tf.name_scope('layer-' + str(depth)):
            W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), 'float', name='weights')
            b = tf.Variable(tf.zeros([output_dim]), 'float', name='biases')

            y = tf.nn.relu(tf.matmul(x, W) + b, name='activation')
            loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            return y, loss_l2

    def resFunc(self, x, input_dim, output_dim, input, orig_dim, depth):
        with tf.name_scope('layer-' + str(depth)):
            W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), 'float', name='weights')
            b = tf.Variable(tf.zeros([output_dim]), 'float', name='biases')

            W_T = tf.Variable(tf.truncated_normal([ orig_dim, output_dim], stddev=0.1), 'float', name='weights_match')

            y = tf.matmul(x, W) + b
            y += tf.matmul(input, W_T)

            loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            return y, loss_l2

    def batch_norm(self, x, input_dim, depth):
        with tf.name_scope('layer-' + str(depth)):
            batch_mean, batch_var = tf.nn.moments(x, [0])
            shift = tf.Variable(tf.zeros([input_dim]), name='shift')
            scale = tf.Variable(tf.ones([input_dim]), name='scale')

            y = tf.nn.batch_normalization(x, batch_mean, batch_var, shift, scale, self.epsilon, name='batch_normalization')

            return y

    def drop(self, x, keep_prob, depth):
        with tf.name_scope("layer-" + str(depth)):

            y = tf.nn.dropout(x, keep_prob, name='dropout')

            return y