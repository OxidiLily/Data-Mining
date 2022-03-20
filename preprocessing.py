import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Numpy merupakan library python digunakan untuk komputasi matriks.
#Matplatlib merupakan library python untuk presentasi data berupa grafik atau plot


dataset = pd.read_csv('Data.csv')

# x = [:,:-1] = pilih semua baris dalam dataset, serta semua kolom kecuali kolom terakhir (Negative indexing pada Python).
# y = [:,-1] =  pilih semua baris, kolom terakhir.
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print()
print("Import Dataset")
print()
print(x)
print()
print(y)
print(100*'=')


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:, 1:3] =imputer.transform(x[:,1:3])

print()
print("Menghilangkan Missing Value (nan)")
print()
print(x)
print()
print(100*'=')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print()
print("Encoding data kategori (Atribut)")
print()
print(x)
print()
print(100*'=')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print()
print("Encoding data kategori (Class / Label)")
print()
print(y)
print()
print(100*'=')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)

print()
print("Membagi dataset ke dalam training set dan test set")
print()
print(x_train)
print()
print(x_test)
print()
print(y_train)
print()
print(y_test)
print()
print(100*'=')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])

print()
print("Feature Scaling")
print()
print(x_train)
print()
print(x_test)
print()
print(100*'=')
