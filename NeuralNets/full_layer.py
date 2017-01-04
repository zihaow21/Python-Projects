import tensorflow as tf


class Layer(object):
    def __init__(self):
        self.epsilon = 1e-4
        self.l2_lambda = 5e-4

    def layer(self, x, input_dim, output_dim, depth):
        with tf.name_scope("layer-" + str(depth)):

            W = tf.Variable(tf.truncated_normal([input_dim, output_dim]), 'float', name='weights')
            b = tf.Variable(tf.zeros([output_dim]), 'float', name='bias')

            y = tf.nn.relu(tf.matmul(x, W) + b, name='activation')
            loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            return y, loss_l2

    def logit(self, x, input_dim, output_dim, depth):
        with tf.name_scope("layer-" + str(depth)):

            W = tf.Variable(tf.truncated_normal([input_dim, output_dim]), 'float', name='weights')
            b = tf.Variable(tf.zeros([output_dim]), 'float', name='bias')

            y = tf.matmul(x, W) + b
            loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            return y, loss_l2

    def sigmoid(self, x, input_dim, output_dim, depth):
        with tf.name_scope("layer-" + str(depth)):

            W = tf.Variable(tf.truncated_normal([input_dim, output_dim]), 'float', name='weights')
            b = tf.Variable(tf.zeros([output_dim]), 'float', name='bias')

            y = tf.nn.sigmoid(tf.matmul(x, W) + b, name='activation')
            loss_l2 = self.l2_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            return y, loss_l2

    def drop(self, x, keep_prob, depth):
        with tf.name_scope("layer-" + str(depth)):

            y = tf.nn.dropout(x, keep_prob, name='dropout')

            return y

    def batch_norm(self, y, input_dim, depth):
        with tf.name_scope("layer-" + str(depth)):

            batch_mean, batch_var = tf.nn.moments(y, [0], name='mean')
            scale = tf.Variable(tf.ones([input_dim]), name='scale')
            shift = tf.Variable(tf.zeros([input_dim]), name='shift')

            y = tf.nn.batch_normalization(y, batch_mean, batch_var, shift, scale, self.epsilon, name='batch_normalization')

            return y
