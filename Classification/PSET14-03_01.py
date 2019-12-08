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
classifier = svm.SVC(C=param_C,gamma=param_gamma)
classifier.fit(X_train, y_train)
print("Displaying support vectors")
print(classifier.support_vectors_)
print("Displayed support vectors")

expected = target_test
predicted = classifier.predict(data_test)
show_some_digits(data_test,predicted,title_text="Predicted {}")
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))



MY_SVM = 0o1
if MY_SVM:
    # clf = svm.BinarySVM(kernel='linear', alg='dual', C=1.0)
    clf = svm.SVC(kernel='linear', C=1.0)
    # clf = svm.MultiSVM(kernel='rbf', C=1.0)
else:
    clf = sk_svm.SVC(kernel='linear', decision_function_shape='ovr')

t1 = time.time()
print(data_train.shape, target_train.shape)
# print type(data_train[0,0])
# print data_train[0]
clf.fit(data_train, target_train)
t2 = time.time()
print ("Training time: ", t2 - t1)
target_predict = clf.predict(data_test)
t3 = time.time()
print ("Predicting time: ", t3 - t2)
print ('predicted classes:', target_predict)

print("\n\nClassification report for classifier %s:\n%s\n"
      % (clf, classification_report(target_test, target_predict)))
print("Confusion matrix:\n%s" % confusion_matrix(target_test, target_predict))


print ('\n\nFor training set!')
# target_predict = [clf.predict(sample) for sample in data_train]
target_predict = clf.predict(data_train)
print("\n\nClassification report for classifier %s:\n%s\n"
      % (clf, classification_report(target_train, target_predict)))
print("Confusion matrix:\n%s" % confusion_matrix(target_train, target_predict))


pca = PCA(n_components=2)
mnist.data = pca.fit_transform(mnist.data)
plt.scatter(mnist.data[:subset,0], mnist.data[:subset,1], marker='o')
plt.scatter(mnist.data[subset:,0], mnist.data[subset:,1], marker='x')
plt.show()

