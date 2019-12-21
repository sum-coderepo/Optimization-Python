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
from sklearn.model_selection import train_test_split
# Features
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import _class_means,_class_cov
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn import metrics as metrics


plt.ion()
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

dirpath = 'C:\\Users\\suagrawa\\Desktop\\Spring_2019_IIIT\\Monsoon 2019\\SMAI Assignments\\Assignment2\\dataset\\IMFDB\\'
X,y = load_data(dirpath)
print(y.shape)
print(X.shape)
N,H,W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]
print(N,H,W,C)
print(y)
ind = np.random.randint(0,y.shape[0],6)
disply_images(X[ind,...],y[ind], row=2,col=3)
X = X.reshape((N,H*W*C))
print(X.shape)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print("X_train shape:",X_train.shape)
print("y_train shape:{}".format(y_train.shape))
y_frame=pd.DataFrame()
y_frame['subject ids']=y_train
y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,8),title="Number of Samples for Each Classes")
n_components=90
pca=PCA(n_components=n_components, whiten=True)
pca.fit(X_train)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)
clf = SVC()
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)

#sys.exit(-1)

def normalize(org_dataset):
	mean_vector = np.mean(org_dataset, axis=0)
	dataset = org_dataset - mean_vector

	return dataset, mean_vector
dataset, mean_vector = normalize(X)



from numpy import linalg as la

def calc_eig_val_vec(dataset):
	cov_mat = np.dot(dataset, dataset.T)
	eig_values, eigen_vectors = la.eig(cov_mat)
	eig_vectors = np.dot(dataset.T, eigen_vectors)
	for i in range(eig_vectors.shape[1]):
		eig_vectors[:, i] = eig_vectors[:, i]/la.norm(eig_vectors[:, i])
	return eig_values.astype(float), eig_vectors.astype(float)

eig_values, eig_vectors = calc_eig_val_vec(dataset)

def pca(eig_values, eig_vectors, k):
	k_eig_val = eig_values.argsort()[-k:][::-1]
	eigen_faces = []

	for i in k_eig_val:
		eigen_faces.append(eig_vectors[:, i])

	eigen_faces = np.array(eigen_faces)

	return eigen_faces

eigen_faces = pca(eig_values, eig_vectors, 400)

def reconstruct_images(eigen_faces, mean_vector):
	org_dim_eig_faces = []

	for i in range(eigen_faces.shape[0]):
		org_dim_eig_faces.append(eigen_faces[i].reshape(32, 32, 3))

	org_dim_eig_faces = np.array(org_dim_eig_faces)

	return org_dim_eig_faces

org_dim_eig_faces = reconstruct_images(eigen_faces, mean_vector)
plt.plot(eig_values[:10])
plt.show()
plt.clf()
plt.show()

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 5, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)

i = 0
for g in gs:
	ax = plt.subplot(g)
	ax.imshow(org_dim_eig_faces[i], cmap = plt.get_cmap("gray"))
	ax.set_xticks([])
	ax.set_yticks([])
	i += 1

plt.show()
plt.clf()