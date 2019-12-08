import numpy as np
from pandas import DataFrame
from matplotlib import pyplot

mu1 = [1,0]
mu2 = [0,1.5]

sigma1 = [[1,1.75],[1.75,1]]
sigma2 = [[1,0.75],[0.75,1]]

set0 = np.append(np.random.multivariate_normal(mu1,sigma1,100),np.zeros((100,1)),axis=1)
set1 = np.append(np.random.multivariate_normal(mu2,sigma2,100),np.ones((100,1)),axis=1)
print("Printing Set0 ----------------------------------------------------------------------------------")
print(set0)
print("Printing Set1 ----------------------------------------------------------------------------------")
#print(set1)
#set2 = np.append(set0, set1)
set2 = np.concatenate((set0, set1), axis=0)
print("Printing Set ----------------------------------------------------------------------------------")
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
