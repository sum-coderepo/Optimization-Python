from mnist import MNIST

import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import sys
#Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from scipy.io import loadmat
from sklearn.datasets import fetch_mldata

# import custom module
from Classification.mnist_helpers import *


# it creates mldata folder in your root project folder
#mnist = fetch_mldata('MNIST original', data_home='C:\\Users\\suagrawa\\scikit_learn_data')
from sklearn.datasets import load_digits
mnist = load_digits()


#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target


# Let's have a look at the random 16 images,
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array

#pick  random indexes from 0 to size of our dataset
#show_some_digits(images,targets)
X_data = images/255.0
Y = targets

#split data to train and test
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)


################ Classifier with good params ###########
# Create a classifier: a support vector classifier

param_C = 10
param_gamma = 0.05
classifier = svm.SVC(C=param_C,gamma=param_gamma)

#We learn the digits on train part
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(X_train, y_train)
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))


########################################################
# Now predict the value of the test
expected = y_test
predicted = classifier.predict(X_test)

show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
