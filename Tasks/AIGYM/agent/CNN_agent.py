import tensorflow as tf
import numpy as np


class CNNAgent(object):
    def __init__(self, image_shape, action_space, batch_size, learning_rate, discount, exploration_rate,
                     exploration_decay_rate):
        self.sess = tf.Session()
        self.image_shape = image_shape
        self.outputsize = action_space.n
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.observation = tf.placeholder('float', [1, 80, 72, 1])
        self.action = tf.placeholder(dtype=tf.int32, shape=[1])
        self.next_observation = tf.placeholder('float', [1, 80, 72, 1])
        self.reward = tf.placeholder('float')
        self.n_actions = action_space.n

        self.next_q = tf.reduce_max(self.cnn_net(self.next_observation))
        self.q = self.cnn_net(self.observation)
        self.policy = tf.argmax(self.q, dimension=1)
        pre_q = tf.slice(self.q[0, :], self.action, [1])
        self.loss = tf.reduce_mean(tf.square(self.reward + self.discount * self.next_q - pre_q))
        self.update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.file_address = '/Users/ZW/Dropbox/current/Python-Projects/Tasks/AIGYM/agent/cnnmodel/TR_CNN_RL.ckpt'
        self.saver = tf.train.Saver()

        self.sess.run([tf.initialize_all_variables()])
        # self.saver.restore(self.sess, self.file_address)

    def cnn_net(self, observation):
        with tf.device('/cpu:0'):
            weights = {
                'w1': tf.Variable(tf.truncated_normal([21, 9, 1, 16], stddev=0.1), 'float'),
                'w2': tf.Variable(tf.truncated_normal([21, 9, 16, 32], stddev=0.1), 'float'),
                'wf': tf.Variable(tf.truncated_normal([5*12*32, 512], stddev=0.1), 'float'),
                'wo': tf.Variable(tf.truncated_normal([512, self.outputsize], stddev=0.1), 'float')
            }

            biases = {
                'b1': tf.Variable(tf.zeros([16]), 'float'),
                'b2': tf.Variable(tf.zeros([32]), 'float'),
                'b3': tf.Variable(tf.zeros([512]), 'float'),
                'b4': tf.Variable(tf.zeros([self.outputsize]), 'float')
            }

            c1 = tf.nn.conv2d(observation, weights['w1'], strides=[1, 1, 1, 1], padding='VALID')
            h1 = tf.nn.relu(c1 + biases['b1'])
            h1_pool = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            c2 = tf.nn.conv2d(h1_pool, weights['w2'], strides=[1, 1, 1, 1], padding='VALID')
            h2 = tf.nn.relu(c2 + biases['b2'])
            h2_pool = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            pool_flat = tf.reshape(h2_pool, [-1, 5*12*32])
            f1 = tf.nn.relu(tf.matmul(pool_flat, weights['wf']) + biases['b3'])

            y = tf.matmul(f1, weights['wo']) + biases['b4']

            return y

    def reset(self):
        self.exploration *= self.exploration_decay_rate

    def act(self, observation):
        if np.random.random_sample() < self.exploration:
            return np.random.randint(0, self.n_actions)
        else:
            res = self.sess.run([self.policy], feed_dict={self.observation: self.sess.run(tf.expand_dims(np.array([observation]), -1))})
            return res[0][0]

    def upd(self, observation, action, new_observation, reward, i_episode):
        self.sess.run([self.update], feed_dict={
            self.observation: self.sess.run(tf.expand_dims(np.array([observation]), -1)),
            self.action: np.array([action]),
            self.next_observation: self.sess.run(tf.expand_dims(np.array([new_observation]), -1)),
            self.reward: np.array([reward])
        })
        if i_episode % 100 == 0:
            self.saver.save(self.sess, self.file_address)
            
if __name__ == "__main__":
        pass
