from itertools import combinations
from distutils.util import strtobool


class Classifier(object):

    def __init__(self):
        self.W = None   # TODO: initialize to some random val
        self.b = 0.

    def train(self, X, y, learning_rate=1e-3, reg=1e-5):
        pass
        # self.W += -learning_rate * grad_W
        # self.b += -learning_rate * grad_b

    def predict(self, X):
        pass
        # return y_pred

    def loss(self, node_vectors, queries, labels):
        pass
        # return loss, grad_W
