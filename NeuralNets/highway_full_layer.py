import tensorflow as tf
import numpy as np


class HighwayLayer(object):
    def layer(self, x, input_dim, output_dim, depth):
        with tf.name_scope('layer-' + str(depth)):
            W_H = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), 'float', name="weights")
            b_H = tf.Variable(tf.zeros([output_dim]), 'float', name='bias')

            W_T = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), 'float', name="weights_transform")
            b_T = tf.Variable(tf.truncated_normal([output_dim], stddev=0.1), 'float', name='bias_transform')

            H = tf.nn.relu(tf.matmul(x, W_H) + b_H, name='activation')
            T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transformation')
            C = tf.sub(1.0, T, name='carry_gate')

            y = tf.add(tf.mul(H, T), tf.mul(x, C), 'output')

            return y
