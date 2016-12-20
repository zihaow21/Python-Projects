import tensorflow as tf


class HighwayLayer(object):
    def __init__(self):
        self.l2_lambda = 5e-4

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

    def logit(self, x, input_dim, output_dim, depth):
        with tf.name_scope("layer-" + str(depth)):

            W = tf.Variable(tf.truncated_normal([input_dim, output_dim]), 'float', name='weights')
            b = tf.Variable(tf.truncated_normal([output_dim]), 'float', name='bias')

            y = tf.matmul(x, W) + b
            loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            return y, loss_l2

    def sigmoid(self, x, input_dim, output_dim, depth):
        with tf.name_scope("layer-" + str(depth)):

            W = tf.Variable(tf.truncated_normal([input_dim, output_dim]), 'float', name='weights')
            b = tf.Variable(tf.truncated_normal([output_dim]), 'float', name='bias')

            y = tf.nn.sigmoid(tf.matmul(x, W) + b, name='activation')
            loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            return y, loss_l2