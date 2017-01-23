# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura, Zihao Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Codes are modified based on the work of above author

import numpy as np
import tensorflow as tf
from Tasks.Chatbot_Practice.dataSerialization import Batch


class ProjectOp:
    """
    Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        :param shape: a tuple(input dim, output dim)
        :param scope: encapsulate variables
        :param dtype: the weights type
        """
        self.scope = scope

        with tf.variable_scope('weights_' + self.scope):
            self.W = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), dtype=dtype)
            self.b = tf.get_variable('bias', shape[1], initializer=tf.constant_initializer(value=0), dtype=dtype)

    def getWeights(self):

        return self.W, self.b

    def deco2vocab(self, X):
        """
        project the output of the decoder into the vocabulary space
        :param X: input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b

class Model:
    """
    implementation of a seq2seq model
    structure: Encoder/Decoder
                2 LSTM layers
    """
    def __init__(self, args, dataSerialization):
        """
        :param args: parameters of the model
        :param data: the data object
        """
        print("Model creation...")

        self.dataSerilization = dataSerialization
        self.args = args
        self.dtype = tf.float32

        # placeholders
        self.encoderInputs = None
        self.decoderInputself = None  # same as the decoder targets but plus the <go> token
        self.decoderTargets = None
        self.decoderWeights = None  #

        # main operators
        self.lossFunc = None
        self.optOp = None
        self.outputs = None # outputs a list of probabilities for each word

        # construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        # TODO: use buckets

        # parameters of sampled softmax (for attention mechanism and a large vocabulary size)
        outputProjection = None
        # Sampled softmax size is less than the vocabulary size
        if 0 < self.args.softmaxSamples < self.dataSerilization.getVocabularySize():
            outputProjection = ProjectOp((self.args.hiddenSize, self.dataSerilization.getVocabularySize()), scope='soft_projectsion', dtype=self.dtype)

            def sampledSoftmax(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])  # add one more dimension, ex. turn [1, 0, 1, 1] to [[1], [0], [1], [1]]

                localW = tf.cast(tf.transpose(outputProjection.W), tf.float32)
                localB = tf.cast(outputProjection.b, tf.float32)
                localInputs = tf.cast(inputs, tf.float32)

                return tf.cast(tf.nn.sampled_softmax_loss(localW, localB, localInputs, labels, self.args.softmaxSamples,
                                                          self.dataSerilization.getVocabularySize()), self.dtype)

        # create rnn cells
        # enco_decoCell = tf.nn.rnn_cell.GRUCell
        enco_decoCell = tf.nn.rnn_cell.LSTMCell(self.args.hiddenSize, state_is_tuple=True)

        if not self.args.test:
            enco_decoCell = tf.nn.dro
            enco_decoCell = tf.nn.rnn_cell.MultiRNNell()