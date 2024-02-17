import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

datasets = pd.read_csv('glass.csv')
dataX = datasets.drop(columns=['Type'])
dataY = datasets.filter(['Type'])
trainX, testX, trainY, testY = train_test_split(dataX, dataY)

svc = LinearSVC().fit(trainX, trainY['Type'].to_numpy())
predY = svc.predict(testX)

print(round(svc.score(trainX, trainY) * 100, 2))
print(classification_report(testY, predY, zero_division=0))
