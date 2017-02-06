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

    def __call__(self, X):
        """
        project the output of the decoder into the vocabulary space
        :param X: input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:
    """
    implementation of a seq2seq model
    structure: Encoder/Decoder by LSTM layers
    """
    def __init__(self, args, dataSerialization):
        """
        :param data: the data object
        """
        self.args = args
        self.dataSerialization = dataSerialization
        self.dtype = tf.float32

        # placeholders
        self.encoderInputs = None
        self.decoderInputs = None  # same as the decoder targets but plus the <go> token
        self.decoderTargets = None
        self.decoderWeights = None  #

        # main operators
        self.lossFunc = None
        self.optOp = None
        self.outputs = None # outputs a list of probabilities for each word

        # construct the graphs
        self.buildNetwork()

        # train model
        self.train()

    def train(self):
        # with tf.Session(
        #     config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS, inter_op_parallelism_threads=NUM_THREADS,
        #                           log_device_placement=False)) as session:

    def buildNetwork(self):

        # parameters of sampled softmax (for attention mechanism and a large vocabulary size)
        outputProjection = None
        # Sampled softmax size is less than the vocabulary size
        if 0 < self.args.softmaxSamples < self.dataSerialization.getVocabularySize():
            outputProjection = ProjectOp((self.args.hiddenSize, self.dataSerialization.getVocabularySize()), scope='soft_projection', dtype=self.dtype)

            def sampledSoftmax(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])  # add one more dimension, ex. turn [1, 0, 1, 1] to [[1], [0], [1], [1]]

                localW = tf.cast(tf.transpose(outputProjection.W), tf.float32)
                localB = tf.cast(outputProjection.b, tf.float32)
                localInputs = tf.cast(inputs, tf.float32)

                return tf.cast(tf.nn.sampled_softmax_loss(localW, localB, localInputs, labels, self.args.softmaxSamples,
                                                          self.dataSerialization.getVocabularySize()), self.dtype)

        # create rnn cells
        # enco_decoCell = tf.nn.rnn_cell.GRUCell
        enco_decoCell = tf.nn.rnn_cell.LSTMCell(self.args.hiddenSize, state_is_tuple=True)

        if not self.args.test:
            enco_decoCell = tf.nn.rnn_cell.DropoutWrapper(enco_decoCell, input_keep_prob=1.0, output_keep_prob=0.5)
        enco_decoCell = tf.nn.rnn_cell.MultiRNNCell([enco_decoCell] * self.args.numLayers, state_is_tuple=True)

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs = [tf.placeholder(tf.int32, [None, ]) for _ in range(self.args.maxLengthEnco)]  # batchsize * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]
            self.decoderTargets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]

        # Define the network
        decoderOutputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs=self.encoderInputs,
                                                                          decoder_inputs=self.decoderInputs,
                                                                          cell=enco_decoCell,
                                                                          num_encoder_symbols=self.dataSerialization.getVocabularySize(),
                                                                          num_decoder_symbols=self.dataSerialization.getVocabularySize(),
                                                                          embedding_size=self.args.embeddingSize,
                                                                          output_projection=outputProjection.getWeights() if outputProjection else None,
                                                                          feed_previous=bool(self.args.test))
        # for tesing
        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]
        # for training
        else:
            self.lossFunc = tf.nn.seq2seq.sequence_loss(decoderOutputs,
                                                        self.decoderTargets,
                                                        self.decoderWeights,
                                                        self.dataSerialization.getVocabularySize(),
                                                        softmax_loss_function=sampledSoftmax if outputProjection else None)

        # Initialize the optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999, epsilon=1e08)
        self.optOp = opt.minimize(self.lossFunc)

    def step(self, batch):
        """
        does not run on itself, just return the operators
        :param ops: a tuple of the (optimization, loss function) for training or (outputs, ) for testing
        :param feedDict: the feedDict for placeholder feeding
        """
        feedDict = {}
        ops = None

        if not self.args.test:  # training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]

            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]] = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFunc)
        else:  # testing
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]] = [self.dataSerialization.goToken]

            ops = (self.outputs, )

        return ops, feedDict