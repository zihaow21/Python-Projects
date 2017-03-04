import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pickle


class Seq2seq(object):
    """
        source_vocab_size: size of the source vocabulary.
        target_vocab_size: size of the target vocabulary.
        hidden_size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        batch_size: the size of the batches used during training;
        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rat e by this much when needed.
        use_lstm: if true, we use LSTM cells instead of GRU cells.
        num_samples: number of samples for sampled softmax.
    """

    def __init__(self, epochs, learning_rate, batch_size, vocab_size, source_vocab_size, target_vocab_size, maxLengthEnco,
                 maxLengthDeco, num_softmax_samples, embedding_size, hidden_size, num_layers, use_lstm, model_dir,
                 meta_dir, word2vec_dict_dir, num_threads, data_object):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_softmax_samples = num_softmax_samples
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.use_lstm = use_lstm
        self.num_layers = num_layers
        self.maxLengthEnco = maxLengthEnco
        self.maxLengthDeco = maxLengthDeco
        self.model_dir = model_dir
        self.meta_dir = meta_dir
        self.word2vec_dict_dir = word2vec_dict_dir
        self.data_object = data_object
        self.num_threads = num_threads

        with open(self.word2vec_dict_dir, 'r') as f:
            self.embedding_dict = pickle.load(f)

        with open(self.word2vec_dict_dir, 'r') as f:
            word2vec_dict = pickle.load(f)

        self.embedding_init = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_size)).astype("float32")
        for word in word2vec_dict.keys():
            word_idx = word2vec_dict[word][1]
            self.embedding_init[word_idx, :] = word2vec_dict[word][0]
        self.embedding_tensor = tf.convert_to_tensor(self.embeddings)

        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_targets = None
        self.decoder_weights = None
        self.output_dropout = None
        self.embeddings = None

        self.loss_func = None
        self.optimizer = None
        self.saver = None

        self.decoder_outputs = None

    def model(self, test=False):
        with tf.name_scope('placeholder_encoder'):
            self.encoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthEnco)]

        with tf.name_scope('placeholder_decoder'):
            self.decoder_inputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.maxLengthDeco)]
            self.decoder_targets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in range(self.maxLengthDeco)]
            self.decoder_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.maxLengthDeco)]

        with tf.name_scope('dropout'):
            self.output_dropout = tf.placeholder('float')

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
            rnnCell = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)
        else:
            rnnCell = tf.contrib.rnn.GRUCell(self.hidden_size)

        rnnCellDropout = tf.contrib.rnn.DropoutWrapper(rnnCell, input_keep_prob=1.0, output_keep_prob=self.output_dropout)
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([rnnCellDropout] * self.num_layers)
        else:
            cell = rnnCellDropout

        self.saver = tf.train.Saver()

        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs =self.encoder_inputs,
                                                                           decoder_inputs=self.decoder_inputs, cell=cell,
                                                                           num_encoder_symbols=self.source_vocab_size,
                                                                           num_decoder_symbols=self.target_vocab_size,
                                                                           embedding_size=self.embedding_size,
                                                                           output_projection=output_projection,
                                                                           feed_previous=test)
        if test:
            if not output_projection:
                self.decoder_outputs = decoderOutputs
            else:
                self.decoder_outputs = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in decoderOutputs]

        else:
            self.loss_func = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, self.decoder_targets, self.decoder_weights,
                                               self.target_vocab_size, softmax_loss_function=softmax_loss_function)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_func)

    def step(self, batch, test=False):
        feed_dict = {}

        if not test:
            for j in range(self.maxLengthEnco):
                feed_dict[self.encoder_inputs[j]] = batch.encoderSeqs[j]

            for k in range(self.maxLengthDeco):
                feed_dict[self.decoder_inputs[k]] = batch.decoderSeqs[k]
                feed_dict[self.decoder_targets[k]] = batch.targetSeqs[k]
                feed_dict[self.decoder_weights[k]] = batch.weights[k]

            feed_dict[self.output_dropout] = 0.5

        else:
            for i in range(self.maxLengthEnco):
                feed_dict[self.encoder_inputs[i]] = batch.encoderSeqs[i]
            feed_dict[self.decoder_inputs[0]] = [self.data_object.goToken]
            feed_dict[self.output_dropout] = 1.0

        return feed_dict

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for i in tqdm(range(self.epochs)):
                batches = self.data_object.getBatches()
                for batch in tqdm(batches):
                    feed_dict = self.step(batch)
                    sess.run(self.optimizer, feed_dict)
                if i % 100 == 0:
                    self.saver.save(sess, self.model_dir)

    def generation(self, questions):
        self.model(test=True)
        answers = []

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.saver.restore(sess, self.model_dir)
            for i, question in enumerate(questions):
                batch = self.data_object.sent2enco(question)
                if batch:
                    feed_dict = self.step(batch)
                    answer_code = sess.run(self.decoder_outputs, feed_dict)
                    answer = self.data_object.vec2str(self.data_object.deco2vec(list(answer_code)))
                    answers.append(answer)
                else:
                    continue
        return answers
