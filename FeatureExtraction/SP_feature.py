from sparse_filtering import SparseFiltering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SP(object):
    def __init__(self, num_feature_extract):
        self.num_feature_extract = num_feature_extract

    def extraction(self, data):
        # extracting 40 features using sparse filtering
        sp = SparseFiltering(self.num_feature_extract)
        feature_set = sp.fit_transform(data)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(feature_set[:, 0], feature_set[:, 1], feature_set[:, 2], c=label[:, 0])
        # ax.set_xlabel('feature 1')
        # ax.set_ylabel('feature 2')
        # ax.set_zlabel('feature 3')
        # plt.title('SP visulization')
        # plt.show()

        return feature_set
