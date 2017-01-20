from tensorflow.examples.tutorials.mnist import input_data
from NeuralNets.fnn import FNN
from NeuralNets.highway_fnn import FNNHighWay
from NeuralNets.residual_fnn import FNNRes
import numpy as np
import matplotlib.pyplot as plt
import time


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

data_train = mnist.train.images
label_train = mnist.train.labels

data_test = mnist.test.images
label_test = mnist.test.labels

data_validate = mnist.validation.images
label_validate = mnist.validation.labels

shape = np.shape(mnist.train.images)
num_samples = shape[0]
input_dim = shape[1]

### tradiional fully connected neural network

model_dir_fnn = '/Users/ZW/Dropbox/current/temp/MNIST_fnn.ckpt'
meta_dir_fnn = '/Users/ZW/Dropbox/current/temp/MNIST_fnn.ckpt.meta'

start_time = time.time()

fcn = FNN(epochs=100, num_layers=4, num_samples=num_samples, input_dim=input_dim, h1_nodes=100, h2_nodes=100, h3_nodes=100,
          num_classes=10, learning_rate=0.001, batch_size=100, model_dir=model_dir_fnn, meta_dir=meta_dir_fnn)

acc_train_fnn, acc_test_fnn, loss_fnn = fcn.classify_train_test(data_train, label_train, data_test, label_test)

end_time = time.time()
duration = end_time - start_time
print "The duration for fully connected neural network is {}".format(duration)

x = np.linspace(1, 20, 20)
plt.figure(1)
plt.plot(x, np.reshape(acc_train_fnn, [20, ]), 'r-')
plt.plot(x, np.reshape(acc_test_fnn, [20, ]), 'g-')
plt.legend(['training acc', 'testing acc'])
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.title('performance of fully connected neural network')

plt.figure(2)
plt.plot(x, np.reshape(loss_fnn, [20, ]), 'b-')
plt.legend(['loss'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('cost of fully connected neural network')

### highway fully connected neural network

model_dir_fnn_highway = '/Users/ZW/Dropbox/current/temp/MNIST_fnn_highway.ckpt'
meta_dir_fnn_highway = '/Users/ZW/Dropbox/current/temp/MNIST_fnn_highway.ckpt.meta'

start_time = time.time()

hfcn = FNNHighWay(epochs=100, num_layers=4, num_samples=num_samples, input_dim=input_dim, h1_nodes=100, h2_nodes=100, h3_nodes=100,
    num_classes=10, learning_rate=0.0005, batch_size=100, model_dir=model_dir_fnn_highway, meta_dir=meta_dir_fnn_highway)

acc_train_fnn_highway, acc_test_fnn_highway, loss_fnn_highway = fcn.classify_train_test(data_train, label_train, data_test, label_test)

end_time = time.time()
duration = end_time - start_time
print "The duration for highway fully connected neural network is {}".format(duration)

plt.figure(3)
plt.plot(x, np.reshape(acc_train_fnn_highway, [20, ]), 'r-')
plt.plot(x, np.reshape(acc_test_fnn_highway, [20, ]), 'g-')
plt.legend(['training acc', 'testing acc'])
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.title('performance of highway fully connected neural network')

plt.figure(4)
plt.plot(x, np.reshape(loss_fnn_highway, [20, ]), 'b-')
plt.legend(['loss'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('cost of highway fully connected neural network')

### residual fully connected neural network

model_dir_fnn_highway = '/Users/ZW/Dropbox/current/temp/MNIST_fnn_res.ckpt'
meta_dir_fnn_highway = '/Users/ZW/Dropbox/current/temp/MNIST_fnn_res.ckpt.meta'

start_time = time.time()

resfcn = FNNRes(epochs=100, num_layers=4, num_samples=num_samples, input_dim=input_dim, h1_nodes=100, h2_nodes=100, h3_nodes=100,
    num_classes=10, learning_rate=0.0005, batch_size=100, model_dir=model_dir_fnn_highway, meta_dir=meta_dir_fnn_highway)

acc_train_fnn_res, acc_test_fnn_res, loss_fnn_res = resfcn.classify_train_test(data_train, label_train, data_test, label_test)

end_time = time.time()
duration = end_time - start_time
print "The duration for residual fully connected neural network is {}".format(duration)

plt.figure(5)
plt.plot(x, np.reshape(acc_train_fnn_res, [20, ]), 'r-')
plt.plot(x, np.reshape(acc_test_fnn_res, [20, ]), 'g-')
plt.legend(['training acc', 'testing acc'])
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.title('performance of residual fully connected neural network')

plt.figure(6)
plt.plot(x, np.reshape(loss_fnn_res, [20, ]), 'b-')
plt.legend(['loss'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('cost of residual fully connected neural network')
plt.show()