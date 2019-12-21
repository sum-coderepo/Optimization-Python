from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

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

X = np.column_stack((x, y))

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

plt.show()