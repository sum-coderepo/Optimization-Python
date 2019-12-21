import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import sys
# generate 2d classification dataset
import numbers
import array
import numpy as np


# read data
X, y = make_blobs(n_samples=2000, centers=2, n_features=2)


# use LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Coefficient of the features in the decision function. (from theta 1 to theta n)
parameters = log_reg.coef_[0]
# Intercept (a.k.a. bias) added to the decision function. (theta 0)
parameter0 = log_reg.intercept_

# Plotting the decision boundary
fig = plt.figure(figsize=(10,7))
x_values = [np.min(X[:, 1] -5 ), np.max(X[:, 1] +5 )]
# calcul y values
y_values = np.dot((-1./parameters[1]), (np.dot(parameters[0],x_values) + parameter0))
colors=['red' if l==0 else 'blue' for l in y]
plt.scatter(X[:, 0], X[:, 1], label='Logistics regression', color=colors)
plt.plot(x_values, y_values, label='Decision Boundary')
plt.show()