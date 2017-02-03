import tensorflow as tf


class ProjectOp(object):
    def __init__(self):
        with tf.variable_scope("projection"):
            self.W = tf.get_variable()