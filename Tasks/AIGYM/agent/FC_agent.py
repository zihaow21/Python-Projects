import tensorflow as tf
import numpy as np

class BNN(object):
    def __init__(self, n_h1, n_h2, input_size, num_classes,  learning_rate, discount,
                 exploration_rate, exploration_decay_rate, action_space):
        self.sess = tf.Session()
        self.num_classes = num_classes  # number of classification classes
        self.n_h1 = n_h1  # number of neurons in the first hidden layer
        self.n_h2 = n_h2  # number of neurons in the second hidden layer
        self.input_size = input_size  # the size of one input data point
        self.file_address = '/Users/ZW/Dropbox/current/Python-Projects/Tasks/AIGYM/agent/fnnmodel/FNN.ckpt'
        self.epsilon = 1e-4


        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.observation = tf.placeholder('float', [1, 80*72])
        self.action = tf.placeholder(dtype=tf.int32, shape=[1])
        self.next_observation = tf.placeholder('float', [1, 80*72])
        self.reward = tf.placeholder('float')
        self.n_actions = action_space.n

        self.next_q = tf.reduce_max(self.fnn(self.next_observation, 0.5))
        self.q = self.fnn(self.observation, 0.5)
        self.policy = tf.argmax(self.q, dimension=1)
        pre_q = tf.slice(self.q[0, :], self.action, [1])
        self.loss = tf.reduce_mean(tf.square(self.reward + self.discount * self.next_q - pre_q))

        self.update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.sess.run([tf.initialize_all_variables()])
        # self.saver.restore(self.sess, self.file_address)



    def fnn(self, observation, dropout):
        """
        Initialization of parameters
        :param weights: initilization of weights for each layer
        :param biases: initilization of biases for each layer
        """
        # input_size is definite/ n_h1, n_h2, n_h3 is self_definite
        # in the environment document, input_size = long array and num_class = 6
        weights = {'w1': tf.Variable(tf.random_normal([self.input_size, self.n_h1])*tf.sqrt(2.0/self.input_size), 'float'),
                   'w2': tf.Variable(tf.random_normal([self.n_h1, self.n_h2])*tf.sqrt(2.0/self.n_h1), 'float'),
                   'w3': tf.Variable(tf.random_normal([self.n_h2, self.num_classes])*tf.sqrt(2.0/self.n_h2), 'float')}

        biases = {'b1': tf.Variable(tf.zeros([self.n_h1]), 'float'),
                  'b2': tf.Variable(tf.zeros([self.n_h2]), 'float'),
                  'b3': tf.Variable(tf.zeros([self.num_classes]), 'float')}

        linear_comb = {}  # linear combination of output from previous layer, weights and biases as input for the next layer
        activations = {}  # activation function applied to the input

        linear_comb['y1'] = tf.add(tf.matmul(observation, weights['w1']), biases['b1'])
        activations['z1'] = tf.nn.relu(linear_comb['y1'])
        h1_dropout = tf.nn.dropout(activations['z1'], dropout)

        linear_comb['y2'] = tf.matmul(h1_dropout, weights['w2'])
        batch_mean1, batch_var1 = tf.nn.moments(linear_comb['y2'], [0])
        z1_bn = (linear_comb['y2'] - batch_mean1) / tf.sqrt(batch_var1 + self.epsilon)
        h2_dropout = tf.nn.dropout(z1_bn, dropout)

        activations['z2'] = tf.nn.relu(h2_dropout)
        output = tf.add(tf.matmul(activations['z2'], weights['w3']), biases['b3'])

        return output

    def reset(self):
        self.exploration *= self.exploration_decay_rate

    def act(self, observation):
        if np.random.random_sample() < self.exploration:
            return np.random.randint(0, self.n_actions)
        else:
            res = self.sess.run([self.policy], feed_dict={self.observation: observation})
            return res[0][0]

    def upd(self, observation, action, new_observation, reward, i_episode):
        self.sess.run([self.update], feed_dict={
            self.observation: observation,
            self.action: np.array([action]),
            self.next_observation: new_observation,
            self.reward: np.array([reward])
        })
        if i_episode % 10 == 1:
             self.saver.save(self.sess, self.file_address)

if __name__ == "__main__":
    pass
