import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

a = np.array([[10, 7, 4], [3, 2, 1]])
print(a)
print(np.percentile(a, 50))
print(np.percentile(a, 50, axis=0))
print(np.percentile(a, 50, axis=1))
m = np.percentile(a, 50, axis=0)
out = np.zeros_like(m)
print(np.percentile(a, 50, axis=0, out=out))
sys.exit(-1)


train_df = pd.read_csv("D://kaggleDatasets//zillow-prize-1//train_2016.csv", parse_dates=["transactiondate"])

ulimit = np.percentile(train_df.logerror.values, 99)
llimit = np.percentile(train_df.logerror.values, 1)
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(train_df.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)

plt.show()