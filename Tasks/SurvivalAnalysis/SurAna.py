import scipy.io as si
import numpy as np
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import sys


from FeatureExtraction.sparse_filtering import SparseFiltering
from FeatureExtraction.PCA_feature import Pca
from NeuralNets.fnn import FNN


# data_file = si.loadmat('/Users/ZW/Downloads/data.mat')
# data = np.transpose(data_file['new_ff'])
# label_file = si.loadmat('/Users/ZW/Downloads/label.mat')
# label = label_file['label']

data = []

# data_file = si.loadmat('/Users/ZW/Downloads/BRCA_version2.mat')
# data = np.transpose(data_file['BRCA_new_features'])
# label = data_file['BRCA_labels']

data_file = si.loadmat('/Users/ZW/Downloads/random_forrest_feature_selection.mat')
data = np.transpose(data_file['rf_results'])
label_file = si.loadmat('/Users/ZW/Downloads/label.mat')
label = label_file['label']

model_dir = '/Users/ZW/Dropbox/current/Python-Projects/Tasks/SurvivalAnalysis/FNN.ckpt'
meta_dir = '/Users/ZW/Dropbox/current/Python-Projects/Tasks/SurvivalAnalysis/FNN.ckpt.meta'

n_slits = 5

for i in range(5):
    ss = ShuffleSplit(n_splits=n_slits, test_size=0.25, random_state=0)

    acc_train = np.zeros([5, 20])
    acc_test = np.zeros([5, 20])

# # define number of features extracted by the feature extraction method
# num_features = 100

# # using features extracted by sparse filtering
# sp = SparseFiltering(num_features)
# data = sp.fit_transform(data)

# # using features extracted by PCA
# pca = Pca(num_features)
# data = pca.extraction(data)

    shape = np.shape(data)
    num_data = shape[0]
    input_dim = shape[1]

    num_samples = np.int(num_data * (1 - 0.25))

    for train_index, test_index in ss.split(data):
        data_train = data[train_index, :].astype(np.float32)
        data_test = data[test_index, :].astype(np.float32)
        label_train = label[train_index, :]
        label_test = label[test_index, :]

        fnn = FNN(epochs=101, num_samples=num_samples, input_dim=input_dim, h1_nodes=200, h2_nodes=200, h3_nodes=200, num_classes=2,
                learning_rate=0.005, batch_size=300, l2_lambda=0.005, model_dir=model_dir, meta_dir=meta_dir)
        acc_n, acc_t = fnn.classify_train_test(data_train, label_train, data_test, label_test)
        acc_train[i, :] = acc_n
        acc_test[i, :] = acc_t

acc_train = np.mean(acc_train)
acc_test = np.mean(acc_test)
x = np.linspace(1, 20, 20)
plt.figure()
plt.plot(x, acc_train, 'r-')
plt.plot(x, acc_test, 'g-')
plt.title('Binary classification accuracy using 2-layer fully connected NN')
plt.legend(['Training', 'Testing'])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
sys.stdout.write('\a')
sys.stdout.flush()
plt.show()

