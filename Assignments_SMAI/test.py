from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import data

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)

    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]

    return (data, labels)

train_data, train_labels = read_data("C:/Users/suagrawa/Desktop/Spring_2019_IIIT/Monsoon 2019/SMAI Assignments/Assignment-1/sample_train.csv")
test_data, test_labels = read_data("C:/Users/suagrawa/Desktop/Spring_2019_IIIT/Monsoon 2019/SMAI Assignments/Assignment-1/sample_test.csv")
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)

muVector = np.mean(train_data, axis=0)
print(muVector)
cov = np.cov(train_data, rowvar=0)
print(cov)

def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.metrics import mean_squared_error

from scipy import interpolate


rs = ShuffleSplit(n_splits=1, train_size = 15, test_size=5)