import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import sys
# generate 2d classification dataset
import numbers
import array
import numpy as np

X, y = make_blobs(n_samples=20, centers=2, n_features=2)
# scatter plot, dots colored by class value
#X : array of shape [n_samples, n_features]
#y : array of shape [n_samples]
print(X)
print(y)

mean1 = np.array([3., 3.])
cov1 = np.array([[3.0, 0.0],[0.0, 3.0]])
set0 = np.append(np.random.multivariate_normal(mean1,cov1,500),np.zeros((500,1)),axis=1)

mean2 = np.array([7., 7.])
cov2 = np.array([[3.0, 0.0],[0.0, 3.0]])
set1 = np.append(np.random.multivariate_normal(mean2,cov2,500),np.ones((500,1)),axis=1)

set = np.concatenate((set0, set1), axis=0)
print(set)
X = set[:,:2]
y = set[:,-1]
print(X)
print(y)

df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()


