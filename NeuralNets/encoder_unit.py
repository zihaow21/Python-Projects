import tensorflow as tf


class Encoder(object):

    def encoder(self, x, input_dim, output_dim):
        W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), 'float', name='weights')
        b = tf.Variable(tf.zeros([output_dim]), 'float', name='biases')

        layer = tf.nn.relu(tf.matmul(x, W) + b)

        return layer