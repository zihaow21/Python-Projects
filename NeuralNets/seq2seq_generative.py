import tensorflow as tf


# class ProjectOp(object):
#     def __init__(self, dim):
#         with tf.name_scope('projection'):
#             self.W = tf.Variable(tf.truncated_normal(dim, stddev=0.1), name='weights')
#             self.b = tf.Variable(tf.zeros([dim[1]]), name='biases')
#
#     def projection(self, x):
#         with tf.name_scope('projection'):
#             target_complete = tf.matmul(x, self.W) + self.b
#
#         return target_complete

class Seq2seq(object):
    """
        source_vocab_size: size of the source vocabulary.
        target_vocab_size: size of the target vocabulary.
        hidden_size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        batch_size: the size of the batches used during training;
        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.
        use_lstm: if true, we use LSTM cells instead of GRU cells.
        num_samples: number of samples for sampled softmax.
    """

    def __init__(self, epochs, learning_rate, batch_size, source_vocab_size, target_vocab_size, maxLengthEnco,
                 maxLengthDeco, num_softmax_samples, embedding_size, hidden_size, num_layers, use_lstm, model_dir,
                 meta_dir):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_softmax_samples = num_softmax_samples
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.use_lstm = use_lstm
        self.num_layers = num_layers
        self.maxLengthEnco = maxLengthEnco
        self.maxLengthDeco = maxLengthDeco
        self.model_dir = model_dir
        self.meta_dir = meta_dir

    def model(self, encoder_inputs, decoder_inputs, output_dropout):
        output_projection = None
        softmax_loss_function = None

        if 0 < self.num_softmax_samples < self.target_vocab_size:
            # outputProjection = ProjectOp([self.hidden_size, self.target_vocab_size])

            with tf.name_scope('sampled softmax'):
                w = tf.Variable(tf.truncated_normal([self.hidden_size, self.target_vocab_size], stddev=0.1), name='weights')
                w_t = tf.transpose(w)
                b = tf.Variable(tf.zeros([self.target_vocab_size]), name='biases')

            output_projection = (w, b)

            def sampledSoftmax(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, self.num_softmax_samples, self.target_vocab_size)

            softmax_loss_function = sampledSoftmax

        if self.use_lstm:
            rnnCell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True)
        else:
            rnnCell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        rnnCellDropout = tf.nn.rnn_cell.DropoutWrapper(rnnCell, input_keep_prob=1.0, output_keep_prob=output_dropout)
        if self.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([rnnCellDropout] * self.num_layers)
        else:
            cell = rnnCellDropout

        decoderOutputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs =encoder_inputs,
                                                                           decoder_inputs=decoder_inputs, cell=cell,
                                                                           num_encoder_symbols=self.source_vocab_size,
                                                                           num_decoder_symbols=self.target_vocab_size,
                                                                           embedding_size=self.embedding_size,
                                                                           output_projection=output_projection,
                                                                           feed_previous=False)

        return decoderOutputs, softmax_loss_function

    def train(self, data_train_encoder_input, data_train_decoder_input, data_train_decoder_output,
              data_test_encoder_input, data_test_decoder_intput, data_test_decoder_output):
        with tf.name_scope('placeholder_encoder'):
            encoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthEnco)]

        with tf.name_scope('placeholder_decoder'):
            decoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthDeco)]
            decoder_targets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in range(self.maxLengthDeco)]
            decoder_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.maxLengthDeco)]

        output_dropout = tf.placeholder('float')

        decoderOutputs, softmax_loss_function = self.model(encoder_inputs, decoder_inputs, output_dropout)
        loss = tf.nn.seq2seq.sequence_loss(decoderOutputs, decoder_targets, decoder_weights, self.target_vocab_size,
                                           softmax_loss_function=softmax_loss_function)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        for