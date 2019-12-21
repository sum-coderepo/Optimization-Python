from utils import softmax_loss,softmax
from plot_utils import *
import numpy as np
import gzip, pickle
from sklearn.datasets import fetch_mldata
import collections
from sklearn.model_selection import train_test_split
import sys

class TwoLayerNeuralNetwork:

    def __init__(self, num_features=784, num_hiddens=1000, num_classes=10):
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes

        # random initialization: create random weights, set all biases to zero
        self.params = {}
        self.params['W1'] = np.random.randn(num_features, num_hiddens) * 0.001
        self.params['W2'] = np.random.randn(num_hiddens,  num_classes) * 0.001
        self.params['b1'] = np.zeros((num_hiddens,))
        self.params['b2'] = np.zeros((num_classes,))

    def forward(self, X):
        # forward step
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # forward step
        h_in = X @ W1 + b1       # hidden layer input
        h = np.maximum(0, h_in)  # hidden layer output (using ReLU)
        scores = h @ W2 + b2     # neural net output

        return scores

    def train_step(self, X, y):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # forward step
        h_in = X @ W1 + b1       # hidden layer input
        h = np.maximum(0, h_in)  # hidden layer output (using ReLU)
        scores = h @ W2 + b2     # neural net output
        #print("scores values is {} ".format(scores))
        # compute loss
        loss, dscores = softmax_loss(scores, y)

        # backward step
        db2 = dscores.sum(axis=0)
        dW2 = h.T @ dscores

        dh = dscores @ W2.T
        dh[h_in < 0] = 0.0
        db1 = dh.sum(axis=0)
        dW1 = X.T @ dh

        gradient = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        return loss, gradient

    def train(self, X_train, y_train, X_valid, y_valid, batch_size=50,
              alpha=0.001, lmbda=0.0001, num_epochs=10):

        m, n = X_train.shape
        num_batches = m // batch_size

        report = "{:3d}: training loss = {:.2f} | validation loss = {:.2f}"

        losses = []
        for epoch in range(num_epochs):
            train_loss = 0.0

            for _ in range(num_batches):
                W1, b1 = self.params['W1'], self.params['b1']
                W2, b2 = self.params['W2'], self.params['b2']

                # select a random mini-batch
                batch_idx = np.random.choice(m, batch_size, replace=False)
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

                # train on mini-batch
                data_loss, gradient = self.train_step(X_batch, y_batch)
                reg_loss = 0.5 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
                train_loss += (data_loss + lmbda * reg_loss)
                losses.append(data_loss + lmbda * reg_loss)

                # regularization
                gradient['W1'] += lmbda * W1
                gradient['W2'] += lmbda * W2

                # update parameters
                for p in self.params:
                    self.params[p] = self.params[p] - alpha * gradient[p]

            # report training loss and validation loss
            train_loss /= num_batches
            valid_loss = softmax_loss(self.forward(X_valid), y_valid, mode='test')
            print(report.format(epoch + 1, train_loss, valid_loss))

        return losses

    def predict(self, X):
        """ Predict labels for input data.
        """
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        """ Predict probabilties of classes for each input data.
        """
        scores = self.forward(X)
        return softmax(scores)


DATA_PATH = 'C:\\Users\\suagrawa\\Optimization-Python\\Classification\\MNIST_data\\mnist.pkl.gz'

with gzip.open(DATA_PATH, 'rb') as f:
    (X_train, y_train), (X_valid, y_valid), (X_test,  y_test) = pickle.load(f, encoding='latin1')

train_filter = np.where((y_train == 0 ) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]
X_valid, y_valid = X_valid[test_filter], y_valid[test_filter]


print('Training data shape:    ', X_train.shape)
print('Training labels shape:  ', y_train.shape)
print('Validation data shape:  ', X_valid.shape)
print('Validation labels shape:', y_valid.shape)
print('Test data shape:        ', X_test.shape)
print('Test labels shape:      ', y_test.shape)

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classes = ['0', '1']
plot_random_samples(X_train, y_train, classes, samples_per_class=10)

#mnist = fetch_mldata('MNIST original',  data_home='C:\\Users\\suagrawa\\scikit_learn_data')
#subset = 700
#mnist.data = np.r_[mnist.data[:subset, :], mnist.data[7000:7000+subset,:]]
#mnist.target = np.r_[mnist.target[:subset], mnist.target[7000:7000+subset]]
#
#print(mnist.data.shape, mnist.target.shape)
#
#data_train, data_test, target_train, target_test = train_test_split(mnist.data, mnist.target, train_size=6.0/7.0)
#X_train1, X_val, y_train1, y_val = train_test_split(data_train, target_train, test_size=0.25, random_state=1)
#print(data_train.shape, target_train.shape)   # print Training Data
#print(data_test.shape, target_test.shape)  # print Test Data
#print(X_train1.shape, y_train1.shape)  # Split Training Data and print
#print(X_val.shape, y_val.shape)  # Split Validation Data and print
mlp = TwoLayerNeuralNetwork(num_hiddens=20)
losses = mlp.train(X_train, y_train, X_valid, y_valid,
                   alpha=0.05, lmbda=0.001, num_epochs=10)
#
#classes = ['0', '1']
#plot_random_samples(data_train, target_train, classes, samples_per_class=10)


#sys.exit(-1)
#mlp = TwoLayerNeuralNetwork(num_hiddens=20,num_classes=2)
#losses = mlp.train(X_train1, y_train1, X_val, y_val, alpha=0.05, lmbda=0.001, num_epochs=2)