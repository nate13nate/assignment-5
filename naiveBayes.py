import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

datasets = pd.read_csv('glass.csv')
dataX = datasets.drop(columns=['Type'])
dataY = datasets.filter(['Type'])
trainX, testX, trainY, testY = train_test_split(dataX, dataY)

gnb = GaussianNB().fit(trainX, trainY['Type'].to_numpy())
predY = gnb.predict(testX)

print(round(gnb.score(trainX, trainY) * 100, 2))
print(classification_report(testY, predY, zero_division=0))
