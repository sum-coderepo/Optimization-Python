from __future__ import division
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import svm as sk_svm
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
import collections
import seaborn as sns; sns.set()
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from Classification.mnist_helpers import *
mnist = fetch_mldata('MNIST original',  data_home='C:\\Users\\suagrawa\\scikit_learn_data')
print(type(mnist.data))
print("shape (full):", mnist.data.shape, mnist.target.shape)
mnist.data = np.array(mnist.data, dtype='float64') / 255.0
# mnist.data /= 255.0
# print mnist.data[0]
print(type(mnist.data[0, 0]))
# mnist.target = mnist.target.astype(int)

# choose 2 classes only
subset = 700
mnist.data = np.r_[mnist.data[:subset, :], mnist.data[7000:7000+subset,:]]
mnist.target = np.r_[mnist.target[:subset], mnist.target[7000:7000+subset]]

print(mnist.data.shape, mnist.target.shape)

data_train, data_test, target_train, target_test = train_test_split(mnist.data, mnist.target, train_size=6.0/7.0)
print(data_train.shape, collections.Counter(target_train))
print(data_test.shape, collections.Counter(target_test))

X_train = data_train
y_train = target_train
print(target_test)
param_C = 10
param_gamma = 0.05
classifier = svm.SVC(kernel='linear', C=param_C,gamma=param_gamma)
classifier.fit(X_train, y_train)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)