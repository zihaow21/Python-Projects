from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class Pca(object):
    def __init__(self, num_feature_extract):
        self.num_feature_extract = num_feature_extract

    def extraction(self, data):
        # extract40 features using PCA
        pca = PCA(self.num_feature_extract)
        pca.fit(data)
        feature_set = pca.transform(data)[:, 0: self.num_feature_extract]

        eigen_value_ratios = pca.explained_variance_ratio_[0: 100]
        x = np.linspace(1, 100, 100)

        plt.figure()
        plt.plot(x, eigen_value_ratios)
        plt.title('PCA eigenvalue ratios')
        plt.xlabel('X')
        plt.ylabel('Eigenvalue Ratio')
        plt.show()

        return feature_set