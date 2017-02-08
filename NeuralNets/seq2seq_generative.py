import tensorflow as tf
from tqdm import tqdm

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
                 meta_dir, num_threads, data_object):
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
        self.data_object = data_object
        self.num_threads = num_threads



    def model(self, encoder_inputs, decoder_inputs, output_dropout):
        output_projection = None
        softmax_loss_function = None

        if 0 < self.num_softmax_samples < self.target_vocab_size:
            with tf.name_scope('sampled_softmax'):
                w = tf.Variable(tf.truncated_normal([self.hidden_size, self.target_vocab_size]), name='weights')
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

    def train(self):
        with tf.name_scope('placeholder_encoder'):
            encoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthEnco)]

        with tf.name_scope('placeholder_decoder'):
            decoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthDeco)]
            decoder_targets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in range(self.maxLengthDeco)]
            decoder_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.maxLengthDeco)]
        with tf.name_scope('dropout'):
            output_dropout = tf.placeholder('float')

        outputs, softmax_loss_func = self.model(encoder_inputs, decoder_inputs, output_dropout)
        loss = tf.nn.seq2seq.sequence_loss(outputs, decoder_targets, decoder_weights, self.target_vocab_size,
                                           softmax_loss_function=softmax_loss_func)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            for i in tqdm(range(self.epochs)):
                batches = self.data_object.getBatches()
                for batch in tqdm(batches):
                    feed_dict = {}
                    for j in range(self.maxLengthEnco):
                        feed_dict[encoder_inputs[j]] = batch.encoderSeqs[j]

                    for k in range(self.maxLengthDeco):
                        feed_dict[decoder_inputs[k]] = batch.decoderSeqs[k]
                        feed_dict[decoder_targets[k]] = batch.targetSeqs[k]
                        feed_dict[decoder_weights[k]] = batch.weights[k]

                        feed_dict[output_dropout] = 0.5

                    sess.run(optimizer, feed_dict=feed_dict)
                if i % 5 == 0:
                    saver.save(sess, self.model_dir)

    def generation(self, question):
        batch = self.data_object.sent2enco(question)

        with tf.name_scope('placeholder_encoder'):
            encoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthEnco)]
        with tf.name_scope('placeholder_decoder'):
            decoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthDeco)]
            decoder_targets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in range(self.maxLengthDeco)]
            decoder_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.maxLengthDeco)]
        with tf.name_scope('dropout'):
            output_dropout = tf.placeholder('float')

        outputs, _ = self.model(encoder_inputs, decoder_inputs, output_dropout)
        new_saver = tf.train.import_meta_graph(self.meta_dir)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
            feed_dict = {}
            for i in range(self.maxLengthEnco):
                feed_dict[encoder_inputs[i]] = batch.encoderSeqs[i]
            feed_dict[decoder_inputs[0]] = [self.data_object.goToken]

        answer_code = sess.run(outputs, feed_dict)
        answer = self.data_object.deco2vec(answer_code)

        return answer
