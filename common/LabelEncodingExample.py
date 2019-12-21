import numpy as np
import pandas as pd
import random
from operator import le, eq
from sklearn.decomposition import PCA
from sklearn import model_selection, preprocessing
df = pd.read_csv("C:\\Users\\suagrawa\\Downloads\\100 Sales Records.csv")
print(df.head(10))
le = preprocessing.LabelEncoder()

cols = ["Region","Country","Item Type"]


def MultiLabelEncoding( InputDF, cols):
    le = preprocessing.LabelEncoder()
    try:
        for col in cols:
            InputDF[col] = le.fit_transform(InputDF[col].values)
        return InputDF
    except Exception as e:
        raise e

df_la = MultiLabelEncoding(df , cols)
print(df_la.head(10))



