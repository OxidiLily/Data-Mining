import numpy as np
import pandas as pd
from sklearn import tree

irisDataset = pd.read_csv('Decision Tree/Dataset Iris.csv', delimiter=';', header=0)
irisDataset.head()

irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]
irisDataset.head()
print(irisDataset)

irisDataset = irisDataset.drop(labels="Id", axis=1)
print(irisDataset)

irisDataset = irisDataset.to_numpy()
print(irisDataset)

dataTraining = np.concatenate((irisDataset[0:40, :], irisDataset[50:90, :]),
                              axis=0)
dataTesting = np.concatenate((irisDataset[40:50, :], irisDataset[90:100, :]),
                              axis=0)

print(dataTesting)
len(dataTesting)

inputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]
model = tree.DecisionTreeClassifier()
model = model.fit(inputTraining, labelTraining)
hasilPrediksi = model.predict(inputTesting)
print("Label sebenarnya : ", labelTesting)
print("Hasil Prediksi : ", hasilPrediksi)


prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("Prediksi benar : ", prediksiBenar, "data")
print("Prediksi salah : ", prediksiSalah, "data")
print("Akurasi : ", prediksiBenar/(prediksiBenar+prediksiSalah)
      *100, "%")