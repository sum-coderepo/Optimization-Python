# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from common.ClassificationMetrics import *
from sklearn.metrics import *
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
print(dataframe.head(10))
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
y_Predict = model.predict(X_test)
objCM = ClassificationMetrics(y_Predict,Y_test)
print('Accuracy', str(objCM.getAccuracyScore()))
print('Precision', str(objCM.getPrecisionScore()))
print('Recall', str(objCM.getRecallScore()))
print('F1 Score', str(objCM.getF1Score()))
#amlLogger.LogMessage('AUC - ', objCM.getROCAUCScore())
filename = 'Logit_model.pkl'
joblib.dump(model, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

print(loaded_model.predict(X_test[1:10]))