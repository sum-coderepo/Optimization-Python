import os
import math
import sys
import importlib

import numpy as np

import pandas as pd

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from scipy.stats import norm

import keras
from keras import backend as bkend
from keras.datasets import cifar10, mnist
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, convolutional, pooling
from keras import metrics

from get_session import get_session
import keras.backend.tensorflow_backend as KTF
#KTF.set_session(get_session(gpu_fraction=0.75, allow_soft_placement=True, log_device_placement=False))

import tensorflow as tf
from tensorflow.python.client import device_lib

from plotnine import *

import matplotlib.pyplot as plt

#from autoencoders_keras.vanilla_autoencoder import VanillaAutoencoder
#from autoencoders_keras.convolutional_autoencoder import ConvolutionalAutoencoder
#from autoencoders_keras.convolutional2D_autoencoder import Convolutional2DAutoencoder
#from autoencoders_keras.seq2seq_autoencoder import Seq2SeqAutoencoder
#from autoencoders_keras.variational_autoencoder import VariationalAutoencoder

np.set_printoptions(suppress=True)

os.environ["KERAS_BACKEND"] = "tensorflow"
importlib.reload(bkend)

print(device_lib.list_local_devices())

mnist = mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist
X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1] * X_train.shape[1]])
X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1] * X_test.shape[1]])
y_train = y_train.ravel()
y_test = y_test.ravel()
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.0
X_test /= 255.0

scaler_classifier = MinMaxScaler(feature_range=(0.0, 1.0))
logistic = linear_model.LogisticRegression(random_state=666)
linear_mod = linear_model.ElasticNetCV()
lb = LabelBinarizer()
lb = lb.fit(y_train.reshape(y_train.shape[0], 1))

pipe_base = Pipeline(steps=[("scaler_classifier", scaler_classifier),
                            ("classifier", logistic)])
pipe_base = pipe_base.fit(X_train, y_train)

acc_base = pipe_base.score(X_test, y_test)
plt.show(acc_base)

print("The accuracy score for the MNIST classification task without autoencoders: %.6f%%." % (acc_base * 100))