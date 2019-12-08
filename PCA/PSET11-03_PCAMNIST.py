import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime

from plyfile import PlyData

from sklearn.decomposition import PCA
""" 
The goal is to reduce the dimensionality of a 3D scan from three to two using PCA to cast a shadow of the data onto its two most 
important principal components. Then render the resulting 2D scatter plot.
The scan is a real life armadillo sculpture scanned using a Cyberware 3030 MS 3D scanner at Stanford University. 
The sculpture is available as part of their 3D Scanning Repository, and is a very dense 3D mesh consisting of 172974 vertices! 
The PLY file is available at https://graphics.stanford.edu/data/3Dscanrep/
"""

# Every 100 data samples, we save 1. If things run too
# slow, try increasing this number. If things run too fast,
# try decreasing it... =)
reduce_factor = 100


# Look pretty...
matplotlib.style.use('ggplot')


# Load up the scanned armadillo
plyfile = PlyData.read('Datasets/stanford_armadillo.ply')
armadillo = pd.DataFrame({
    'x':plyfile['vertex']['z'][::reduce_factor],
    'y':plyfile['vertex']['x'][::reduce_factor],
    'z':plyfile['vertex']['y'][::reduce_factor]
})


def do_PCA(armadillo):
    #
    # import the libraries required for PCA.
    # Then, train your PCA on the armadillo dataframe. Finally,
    # drop one dimension (reduce it down to 2D) and project the
    # armadillo down to the 2D principal component feature space.
    #

    pca = PCA(n_components=2)
    pca.fit(armadillo)
    PCA(copy=True, n_components=2, whiten=False)

    reducedArmadillo = pca.transform(armadillo)

    return reducedArmadillo


def do_RandomizedPCA(armadillo):
    #
    # import the libraries required for
    # RandomizedPCA. Then, train your RandomizedPCA on the armadillo
    # dataframe. Finally, drop one dimension (reduce it down to 2D)
    # and project the armadillo down to the 2D principal component
    # feature space.
    #
    #
    # NOTE: SKLearn deprecated the RandomizedPCA method, but still
    # has instructions on how to use randomized (truncated) method
    # for the SVD solver. To find out how to use it, check out the
    # full docs here:
    # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #

    pca = PCA(n_components=2, svd_solver='randomized')
    pca.fit(armadillo)
    PCA(copy=True, n_components=2, whiten=False)

    reducedArmadillo = pca.transform(armadillo)

    return reducedArmadillo


# Render the Original Armadillo
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Armadillo 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(armadillo.x, armadillo.y, armadillo.z, c='green', marker='.', alpha=0.75)



# Time the execution of PCA 5000x
# PCA is ran 5000x in order to help decrease the potential of rogue
# processes altering the speed of execution.
t1 = datetime.datetime.now()
for i in range(5000): pca = do_PCA(armadillo)
time_delta = datetime.datetime.now() - t1

# Render the newly transformed PCA armadillo!
if not pca is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('PCA, build time: ' + str(time_delta))
    ax.scatter(pca[:,0], pca[:,1], c='blue', marker='.', alpha=0.75)



# Time the execution of rPCA 5000x
t1 = datetime.datetime.now()
for i in range(5000): rpca = do_RandomizedPCA(armadillo)
time_delta = datetime.datetime.now() - t1

# Render the newly transformed RandomizedPCA armadillo!
if not rpca is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('RandomizedPCA, build time: ' + str(time_delta))
    ax.scatter(rpca[:,0], rpca[:,1], c='red', marker='.', alpha=0.75)


plt.show()