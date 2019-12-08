import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header=None)
print(df_wine.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

covariant_matrix = np.cov(X_train_std.T)
covariant_matrix[0::5]
eigen_values, eigen_vectors = np.linalg.eig(covariant_matrix)
eigen_values, eigen_vectors[::5]
idx = eigen_values.argsort()[::-1]
eigenValues_sort = eigen_values[idx]
print("printing Eigen values in sorted order")
print(eigenValues_sort)
print("---------------------------------------------------------------------------")

tot = sum(eigen_values)
var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
    label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()
eigen_pairs = \
    [(np.abs(eigen_values[i]),eigen_vectors[:,i]) for i in range(len(eigen_values))]
eigen_pairs[:5]
w= np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
w.shape
print(w)
print(X_train_std[0])
print(X_train_std[0].dot(w))
X_train_pca = X_train_std.dot(w)
X_train_std.shape, w.shape, X_train_pca.shape
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1],
            c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()