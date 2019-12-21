# Basic Imports
import os
import sys
import warnings
import numpy as  np
import pandas as pd
from scipy import linalg

# Loading and plotting data
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Features
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import _class_means,_class_cov
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

opt = {
    'image_size': 32,
    'is_grayscale': False,
    'val_split': 0.75
}

cfw_dict = {'Amitabhbachan': 0,
            'AamirKhan': 1,
            'DwayneJohnson': 2,
            'AishwaryaRai': 3,
            'BarackObama': 4,
            'NarendraModi': 5,
            'ManmohanSingh': 6,
            'VladimirPutin': 7}

imfdb_dict = {'MadhuriDixit': 0,
              'Kajol': 1,
              'SharukhKhan': 2,
              'ShilpaShetty': 3,
              'AmitabhBachan': 4,
              'KatrinaKaif': 5,
              'AkshayKumar': 6,
              'Amir': 7}

# Load Image using PIL for dataset
def load_image(path):
    im = Image.open(path).convert('L' if opt['is_grayscale'] else 'RGB')
    im = im.resize((opt['image_size'],opt['image_size']))
    im = np.array(im)
    im = im/256
    return im

# Load the full data from directory
def load_data(dir_path):
    image_list = []
    y_list = []

    if "CFW" in dir_path:
        label_dict = cfw_dict

    elif "yale" in dir_path.lower():
        label_dict = {}
        for i in range(15):
            label_dict[str(i+1)] = i
    elif "IMFDB" in dir_path:
        label_dict = imfdb_dict
    else:
        raise KeyError("Dataset not found.")


    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".png"):
            im = load_image(os.path.join(dir_path,filename))
            y = filename.split('_')[0]
            y = label_dict[y]
            image_list.append(im)
            y_list.append(y)
        else:
            continue

    image_list = np.array(image_list)
    y_list = np.array(y_list)

    print("Dataset shape:",image_list.shape)

    return image_list,y_list

# Display N Images in a nice format
def disply_images(imgs,classes,row=1,col=2,w=64,h=64):
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, col*row +1):
        img = imgs[i-1]
        fig.add_subplot(row, col, i)

        if opt['is_grayscale']:
            plt.imshow(img , cmap='gray')
        else:
            plt.imshow(img)

        plt.title("Class:{}".format(classes[i-1]))
        plt.axis('off')
    plt.show()

dirpath = 'C:\\Users\\suagrawa\\Desktop\\Spring_2019_IIIT\\Monsoon 2019\\SMAI Assignments\\Assignment2\\dataset\\IMFDB'
X,y = load_data(dirpath)
print(y.shape)
print(X.shape)
N,H,W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]
print(N,H,W,C)
print(y)

ind = np.random.randint(0,y.shape[0],6)
disply_images(X[ind,...],y[ind], row=2,col=3)
# Flatten to apply PCA/LDA
X = X.reshape((N,H*W*C))
print(X.shape)

n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X)
#plt.plot(pca.singular_values_)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance')
#plt.show()
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow((pca.components_[i].reshape(32, 32,3).astype('uint8')), cmap='bone')
plt.show()
