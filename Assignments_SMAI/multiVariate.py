import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "C:\\Users\\suagrawa\\Optimization-Python\\Data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def estimateGaussian(X):
    mu = X.mean()

    m, n = X.shape # number of training examples, number of features

    sigma = np.zeros((n,n))

    for i in range(0,m):
        sigma = sigma + (X.iloc[i] - mu).values.reshape(n,1).dot((X.iloc[i] - mu).values.reshape(1, n))

    sigma = sigma * (1.0/m) # Use 1.0 instead of 1 to force float conversion

    return mu, sigma

def multivariateGaussian(X, mu, sigma):
    m, n = X.shape # number of training examples, number of features

    X = X.values - mu.values.reshape(1,n) # (X - mu)

    # vectorized implementation of calculating p(x) for each m examples: p is m length array
    p = (1.0 / (math.pow((2 * math.pi), n / 2.0) * math.pow(np.linalg.det(sigma),0.5))) * np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(sigma)) * X, axis=1))

    return p

def selectThreshold(yval, pval):
    yval = np.squeeze(yval.values).astype(int)

    bestEpsilon = 0.0
    bestF1 = 0.0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        predictions = (pval < epsilon).astype(int)

        tp = np.sum((predictions == 1).astype(int) & (yval == 1).astype(int))
        fp = np.sum((predictions == 1).astype(int) & (yval == 0).astype(int))
        fn = np.sum((predictions == 0).astype(int) & (yval == 1).astype(int))

        # calculate precision & recall
        prec = (tp * 1.0) / (tp + fp)
        rec = (tp * 1.0) / (tp + fn)

        F1 = (2 * prec * rec) * 1.0 / (prec + rec) # calculate F1 score using current epsilon

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

def main():

    data = pd.read_csv("C:\\Users\\suagrawa\\Optimization-Python\\Data\\creditcard.csv")

    # Group positive and negative examples
    negData = data.groupby('Class').get_group(0)
    posData = data.groupby('Class').get_group(1)

    # Give 60:20:20 split of negative examples for train, validate, test
    train, negCV, negTest = np.split(negData.sample(frac=1), [int(.6*len(negData)), int(.8*len(negData))])

    # Give 50:50 split of positive exampels for validate, test
    posCV, posTest = np.split(posData.sample(frac=1), 2)

    # Concatenate to form final cv and test set
    cv = negCV.append(posCV)
    test = negTest.append(posTest)

    Xtrain = train[train.columns[0:30]]
    ytrain = train[train.columns[30:]]
    XCV = cv[cv.columns[0:30]]
    yCV = cv[cv.columns[30:]]
    Xtest = test[test.columns[0:30]]
    ytest = test[test.columns[30:]]

    print ("Finished splitting data...")

    # Get parameters of gaussian distribution for every feature in Xtrain
    # mu is mean of dataset, and sigma is covariance matrix
    mu, sigma = estimateGaussian(Xtrain)

    print ("Learned mu and sigma...")

    # ptrain = multivariateGaussian(Xtrain, mu, sigma)

    pCV = multivariateGaussian(XCV, mu, sigma)

    print ("Calculated p(x)...")

    epsilon, F1 = selectThreshold(yCV, pCV)

    print ("Found best epsilon = " + str(epsilon) + ", best F1 = " + str(F1))

    ptest = multivariateGaussian(Xtest, mu, sigma) # Fit final model on test set

    predictions = (ptest < epsilon).astype(int)
    ytest = np.squeeze(ytest.values).astype(int)

    tp = np.sum((predictions == 1).astype(int) & (ytest == 1).astype(int))
    fp = np.sum((predictions == 1).astype(int) & (ytest == 0).astype(int))
    fn = np.sum((predictions == 0).astype(int) & (ytest == 1).astype(int))
    tn = np.sum((predictions == 0).astype(int) & (ytest == 0).astype(int))

    prec = (tp * 1.0) / (tp + fp)
    rec = (tp * 1.0) / (tp + fn)

    print ("Precision = " + str(prec) + ", Recall = " + str(rec))


if __name__ == "__main__":
    main()