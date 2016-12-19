import tensorflow as tf
import numpy as np


class Layer(object):
    def __init__(self):
        self.epsilon = 1e-4

    def layer(self, x, input_dim, output_dim, depth):
        with tf.name_scope("layer-" + str(depth)):
            W = tf.Variable(tf.truncated_normal([input_dim, output_dim]), 'float', name='weights')
            b = tf.Variable(tf.truncated_normal([output_dim]), 'float', name='bias')

            y = tf.nn.relu(tf.matmul(x, W) + b, name='activation')

            return y

    def drop(self, y, keep_prob, depth):
        with tf.name_scope("layer-" + str(depth)):
            y_dropout = tf.nn.dropout(y, keep_prob, name='dropout')

            return y_dropout

    def batch_norm(self, y, input_dim, depth):
        with tf.name_scope("layer-" + str(depth)):
            batch_mean, batch_var = tf.nn.moments(y, [0], name='mean')
            scale = tf.Variable(tf.ones([input_dim]), name='scale')
            shift = tf.Variable(tf.zeros([input_dim]), name='shift')

            y_batch_norm = tf.nn.batch_normalization(y, batch_mean, batch_var, shift, scale, self.epsilon, name='batch_normalization')

            return y_batch_norm
