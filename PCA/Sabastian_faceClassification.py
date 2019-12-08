import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import pandas as pd
from glob import iglob
faces = pd.DataFrame([])
for path in iglob('C:\\Users\\suagrawa\\Desktop\\Spring_2019_IIIT\\Monsoon 2019\\SMAI Assignments\\Assignment2\\dataset\\IMFDB\\*.png'):
    img=imread(path)
    face = pd.Series(img.flatten(),name=path)
    faces = faces.append(face)
plt.subplot

fig, axes = plt.subplots(10,10,figsize=(9,9),
                         subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.iloc[i].values.reshape(32,32,3),cmap="gray")

from sklearn.decomposition import PCA
#n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
faces_pca = PCA(n_components=0.8)
faces_pca.fit(faces)
fig, axes = plt.subplots(2,10,figsize=(9,3),
                         subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_pca.components_[i].reshape(112,92),cmap="gray")

components = faces_pca.transform(faces)
projected = faces_pca.inverse_transform(components)
fig, axes = plt.subplots(10,10,figsize=(9,9), subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(projected[i].reshape(112,92),cmap="gray")