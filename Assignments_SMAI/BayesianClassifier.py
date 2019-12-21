import numpy as np
import pylab as pl
from sklearn import mixture

np.random.seed(0)
#C1 = np.array([[3, -2.7], [1.5, 2.7]])
#C2 = np.array([[1, 2.0], [-1.5, 1.7]])
#
#X_train = np.r_[
#    np.random.multivariate_normal((-7, -7), C1, size=7),
#    np.random.multivariate_normal((7, 7), C2, size=7),
#]

X_train = np.r_[
    np.array([[0,0],[0,1],[2,0],[3,2],[3,3],[2,2],[2,0]]),
    np.array([[7,7],[8,6],[9,7],[8,10],[7,10],[8,9],[7,11]]),
    ]

print(X_train)

clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.weights_ = [2,1]
clf.fit(X_train)

#define g1(x, y) and g2(x, y)

def g1(x, y):
    print("x = {},y = {} for g1".format(x,y))
    return clf.predict_proba(np.column_stack((x, y)))[:, 0]

def g2(x, y):
    print("x = {},y = {} for g2".format(x,y))
    return clf.predict_proba(np.column_stack((x, y)))[:, 1]

X, Y = np.mgrid[-15:13:500j, -15:13:500j]
x = X.ravel()
y = Y.ravel()

p = (g1(x, y) - g2(x, y)).reshape(X.shape)

pl.scatter(X_train[:, 0], X_train[:, 1])
pl.contour(X, Y, p, levels=[0])
pl.show()