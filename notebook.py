# -*- coding: utf-8 -*-
"""notebook.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t1AucAhycIYwu0Y9lw9eSzSGfUlK-zQC
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

# !kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009
# !unzip red-wine-quality-cortez-et-al-2009.zip

"""### Data Understanding"""

# load the dataset
import pandas as pd
df = pd.read_csv("winequality-red.csv")
df

df.info()

df.describe()

# Cek tipe data numerik
Integertipe = (df.dtypes == 'int64')
floattipe = (df.dtypes == 'float64')
NumericVariablesnya = list(Integertipe[Integertipe].index) + list(floattipe[floattipe].index)

## Cek data yang berjenis numerik
NumericVariablesnya

# Cek tipe data kategorikal/nominal
nominal= (df.dtypes == 'object')
CategoricalVariablesnya = list(nominal[nominal].index)

## Cek data yang berjenis kategori
CategoricalVariablesnya

"""### Data Preparation"""

df.isna().sum()

df.duplicated().sum()

df.drop_duplicates()

"""#### Outlier"""

numeric_columns = df.columns
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR=Q3-Q1
df = df.loc[~((df[numeric_columns]<(Q1-1.5*IQR))|(df[numeric_columns]>(Q3+1.5*IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
df.shape

"""#### Univariate Analysis"""

df.hist(bins=50, figsize=(20,15))
plt.show()

df['fixed acidity'].value_counts()

df['volatile acidity'].value_counts()

df['citric acid'].value_counts()

df['residual sugar'].value_counts()

df['chlorides'].value_counts()

df['free sulfur dioxide'].value_counts()

df['total sulfur dioxide'].value_counts()

df['density'].value_counts()

df['pH'].value_counts()

df['sulphates'].value_counts()

df['alcohol'].value_counts()

df['quality'].value_counts()

"""#### Multivariate Analysis"""

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

# Cek korelasi antara target dan predictor
kolsnum = df.select_dtypes(include=['int64', 'float64']).columns
korelasi = df[kolsnum].corr()['quality'].sort_values(ascending=False)
korelasi

kolsnum2 = df.select_dtypes(include=['int64', 'float64']).columns
korelasi2 = df[kolsnum2].corr()['free sulfur dioxide'].sort_values(ascending=False)
korelasi2

"""`Volatile acidity` memiliki korelasi negatif, yang artinya jika semakin bagus kualitas dari wine tersebut (mendekati 10), maka tingkat `volatile aciditynya` akan semakin rendah. Dengan kata lain masih memiliki hubungan meskipun tegak lurus

Disisi lain perhatikan korelasi yang dimiliki atribut `free sulfur dioxodie`. Yang menunjukkan hampir memiliki korelasi yang kecil dengan sebagian besar atribut. Maka dari itu akan kita lakukan feature selection dengan atribut ini

#### Data Selection / Feature Selection
"""

df.drop(['free sulfur dioxide'], inplace=True, axis=1)
df.head()

"""## Modelling"""

# Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

X = df.drop('quality', axis = 1)
y = df['quality']

# # Normalisasi
# X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

print(f'Total # of sample in whole dataset: {len(X)}')

from sklearn.model_selection import train_test_split
# Split data 80 : 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape

print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# # Buat objek scaler
# scaler = StandardScaler()
# # Sesuaikan scaler dengan data
# X_train = scaler.fit_transform(X_train)
# # Mengubah data train dan test
# X_test = scaler.transform(X_test)
# y_train = y_train.to_numpy()
# y_test = y_test.to_numpy()

"""KNN"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""RF"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor

# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""Adaboost"""

from sklearn.ensemble import AdaBoostRegressor

adaboost = AdaBoostRegressor(learning_rate=0.05, random_state=55)
adaboost.fit(X_train, y_train)
models.loc['train_mse','Adaboost'] = mean_squared_error(y_pred=adaboost.predict(X_train), y_true=y_train)

"""DT"""

# # Impor library yang dibutuhkan
# from sklearn.tree import DecisionTreeRegressor

# # buat model prediksi
# dtr = DecisionTreeRegressor(max_depth=16, random_state=55)
# dtr.fit(X_train, y_train)

# models.loc['train_mse','DecisionTree'] = mean_squared_error(y_pred=dtr.predict(X_train), y_true=y_train)

"""Gradient Boosting"""

from sklearn.ensemble import GradientBoostingRegressor

# buat model prediksi
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

models.loc['train_mse','GradientBoosting'] = mean_squared_error(y_pred=gbr.predict(X_train), y_true=y_train)

"""SVR"""

from sklearn.svm import SVR

# buat model prediksi
svr = SVR()
svr.fit(X_train, y_train)

models.loc['train_mse','SVR'] = mean_squared_error(y_pred=svr.predict(X_train), y_true=y_train)

"""### Evaluasi Model"""

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Adaboost','GradientBoosting','SVR'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Adaboost': adaboost, 'GradientBoosting': gbr, 'SVR':svr}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[0:5].copy()
pred_dict = {'y_true':y_test[0:5]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)