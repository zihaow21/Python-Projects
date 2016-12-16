import gym
import numpy as np
import matplotlib.pyplot as plt
from Tasks.AIGYM.agent.CNN_agent import CNNAgent
from Tasks.AIGYM.agent.FC_agent import BNN
from tqdm import tqdm

# def imageinfo(x):
#     # figure out the image information
#     for j in range(210):
#         for k in range(160):
#             if x[j, k, 0] == 142 and x[j, k, 1] == 142 and x[j, k, 2] == 142:
#                 print "The grey is {}, {}".format(j, k)
#             if x[j, k, 0] == 200 and x[j, k, 1] == 72 and x[j, k, 2] == 72:
#                 print "The red is {}, {}".format(j, k)
#     """
#     0-14 rows: score display
#     17-31 rows: grey areas from pixel 0 to pixel 159, the length is 160
#     32-188 rows: play area from pixel 8 to pixel 151, the length is 144
#     189-209 rows: black area
#     32-192 rows and 8-151 columns are effective learning area
#     The shape is 80x72 after resampling
#     """

WINDOW = 10
episodes = 10000

def image_preprocess(x):
    x = x[32: 192, 8: 151, :]
    x_downsample = x[::2, ::2, 0]
    x_downsample[x_downsample != 0] = 1

    return x_downsample

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    # agent = CNNAgent([80, 72], env.action_space, learning_rate=0.0005, batch_size=10, discount=0.9, exploration_rate=0.9, exploration_decay_rate=0.99)
    agent = BNN(n_h1=500, n_h2=500, input_size=80*72, num_classes=6,  learning_rate=0.00005, discount=0.9, exploration_rate=0.9, exploration_decay_rate=0.95, action_space=env.action_space)
    total_rewards = []
    total_lengths = []
    for i_episode in tqdm(range(episodes)):
        observation = env.reset()
        observation = image_preprocess(observation)

        # just for fully connected neural network
        observation = np.reshape(observation, [1, 80*72])

        agent.reset()
        total_reward = 0
        length = 0
        for j in range(100):
            env.render()
            action = agent.act(observation)
            new_observation, reward, done, _ = env.step(action)
            total_reward += reward
            new_observation = image_preprocess(new_observation)

            # just for fully connected neural network
            new_observation = np.reshape(new_observation, [1, 80*72])

            agent.upd(observation, action, new_observation, reward, i_episode)
            observation = new_observation
            length += 1
            if done:
                break

        total_rewards.append(total_reward)
        total_lengths.append(length)
    plt.plot(np.convolve(total_lengths, np.ones((WINDOW,)) / WINDOW, mode='valid'))
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.show()
