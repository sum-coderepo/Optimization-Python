from sklearn.datasets import make_regression
from matplotlib import pyplot
## generate regression dataset
#X, y = make_regression(n_samples=20, n_features=1, noise=5)
## plot regression dataset
#pyplot.scatter(X,y)
#pyplot.show()

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import matplotlib.pyplot as plt
import numpy as np
#from generate_dataset import generate_dataset as gd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def generate_dataset_simple(beta, n, std_dev , xax):
    x = np.random.random_sample(n) * 100
    e = np.random.randn(n) * std_dev
    y = x * beta + e + xax
    # We need to reshape x to be a 2-d matrix with n rows and 1 column
    # This is so that it can take a generalized form that can be expanded
    # to multiple predictors for the `LinearRegression` model
    return x.reshape(n, 1), y

beta = 10
x1, y1 = generate_dataset_simple(beta, 500, 50, 0)

x2,y2 = generate_dataset_simple(beta, 500, 50, 1000)

x = np.append(x1,x2)
y = np.append(y1,y2)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.show()