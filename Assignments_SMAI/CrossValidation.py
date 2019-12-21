import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('seaborn-white')
#%matplotlib inline

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.metrics import mean_squared_error

from scipy import interpolate

degs = np.arange(0,11)
degrees = [4]
Train_MSE_list, Test_MSE_list = [], []

#Initializing noisy non linear data
n = 20000
x = np.linspace(0,1,n)
x_plot = np.linspace(0,1,10*n)
noise = np.random.uniform(-.5,.5, size = n)
#y = np.sin(x * 1 * np.pi  - .5)
y = np.sin(x)
y_noise = y + noise
Y = (y + noise).reshape(-1,1)
X = x.reshape(-1,1)

rs = ShuffleSplit(n_splits=1, train_size = 100, test_size=5)
rs.get_n_splits(X)

for train_index, test_index in rs.split(X):
    X_train, X_test, y_train, y_test = X[train_index],X[test_index],Y[train_index], Y[test_index]


#Setup plot figures
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(1, 2, 1)

for d in degs:
    #Create an sklearn pipeline, fit and plot result
    pipeline = Pipeline([('polynomialfeatures', PolynomialFeatures(degree=d, include_bias=True, interaction_only=False)),
                         ('linearregression', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True))])

    pipeline.fit(X_train,y_train)

    Train_MSE = mean_squared_error(y_train, pipeline.predict(X_train))
    Test_MSE = mean_squared_error(y_test, pipeline.predict(X_test))
    Train_MSE_list.append(Train_MSE)
    Test_MSE_list.append(Test_MSE)

    if d in degrees:
        plt.plot(x_plot, pipeline.predict(x_plot.reshape(-1,1)), label = 'd = {}'.format(d), color = 'red')


    #First plot left hand side
ax.plot(x,y,color = 'darkblue',linestyle = '--', label = 'f(x)')
ax.scatter(X_train,y_train, facecolors = 'none', edgecolor = 'darkblue')
ax.set_title('Noisy sine curve, 100 data points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-1.5,1.5)
ax.legend()

#========================== RHS plot ====================#


rs = ShuffleSplit(n_splits=50, train_size = 10000, test_size=15)
rs.get_n_splits(X)

for train_index, test_index in rs.split(X):
    X_train, X_test, y_train, y_test = X[train_index],X[test_index],Y[train_index], Y[test_index]


ax = fig.add_subplot(1, 2, 2)

for d in degs:
    #Create an sklearn pipeline, fit and plot result
    pipeline = Pipeline([('polynomialfeatures', PolynomialFeatures(degree=d, include_bias=True, interaction_only=False)),
                         ('linearregression', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True))])

    pipeline.fit(X_train,y_train)

    Train_MSE = mean_squared_error(y_train, pipeline.predict(X_train))
    Test_MSE = mean_squared_error(y_test, pipeline.predict(X_test))
    Train_MSE_list.append(Train_MSE)
    Test_MSE_list.append(Test_MSE)

    if d in degrees:
        plt.plot(x_plot, pipeline.predict(x_plot.reshape(-1,1)), label = 'd = {}'.format(d), color = 'red')


    #First plot left hand side
ax.plot(x,y,color = 'darkblue',linestyle = '--', label = 'f(x)')
ax.scatter(X_train,y_train, facecolors = 'none', edgecolor = 'darkblue')
ax.set_title('Noisy sine curve, 10000 data points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-1.5,1.5)
ax.legend()

plt.show()


#Utility variables
CV_Mean_MSE_small, CV_Var_MSE_small = [],[]
k_folds_range = np.array([2,4,6,8,10,15,20,25,29,35,39])

for k in k_folds_range:
    #Reset list at start of loop
    i_Mean_MSE = []

    #Repeat experiment i times
    for i in range(300):
        #Reset list at start of loop
        Kfold_MSE_list = []

        #Resample with replacement from original dataset
        rs = ShuffleSplit(n_splits=1, train_size = 100, test_size=1)
        rs.get_n_splits(X)
        for subset_index, _ in rs.split(X):
            X_subset, Y_subset, = X[subset_index],Y[subset_index]

        #Loop over kfold splits
        kf = KFold(n_splits = k)
        for train_index, test_index in kf.split(X_subset):
            X_train, X_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = Y_subset[train_index], Y_subset[test_index]

            #Fit model on X_train
            pipeline = Pipeline([('polynomialfeatures', PolynomialFeatures(degree=4, include_bias=True, interaction_only=False)),
                                 ('linearregression', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True))])
            pipeline.fit(X_train,y_train)

            #Store each Kfold MSE values on X_test
            Kfold_MSE_list.append(mean_squared_error(y_test, pipeline.predict(X_test)))

        #Average over the K folds for a single "i" iteration
        i_Mean_MSE.append(np.mean(Kfold_MSE_list))

    #Average and std for a particular k value over all i iterations
    CV_Mean_MSE_small.append(np.mean(i_Mean_MSE))
    CV_Var_MSE_small.append(np.var(i_Mean_MSE, ddof = 1))

#Convert to numpy for convenience
CV_Mean_MSE_small  = np.asarray(CV_Mean_MSE_small)
CV_Var_MSE_small  = np.asarray(CV_Var_MSE_small)
CV_Std_MSE_small = np.sqrt(CV_Var_MSE_small)

fig = plt.figure(figsize=(16,8))
fig.add_subplot(1, 2, 1)
k_folds_range = np.array([2,4,6,8,10,15,20,25,30,35,39])

plt.fill_between(k_folds_range, 1 - (CV_Mean_MSE_small - CV_Std_MSE_small),
                 1 - (CV_Mean_MSE_small + CV_Std_MSE_small), alpha=0.1, color="g", label = '$\pm 1$ std')

plt.plot(k_folds_range, 1 - CV_Mean_MSE_small, 'o-', color="g",
         label="Cross-validation mean")

plt.hlines(1 - 1/12 , min(k_folds_range),max(k_folds_range), linestyle = '--', color = 'gray', alpha = .5, label = 'True noise $\epsilon$')
plt.legend(loc="lower right"),
plt.ylim(0.7,1)
plt.ylabel('1 - MSE'), plt.xlabel('Kfolds')
plt.title('1 - MSE vs Number of Kfolds: 100 data points, 300 iterations bootstrap ')
