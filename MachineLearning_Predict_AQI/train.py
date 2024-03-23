import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
file_path = 'historical_air_quality_2021_en.csv'
df = pd.read_csv(file_path, index_col = None)
df = df.rename(columns={"AQI index":"AQI"})
df_copy = df.copy(deep = True)
df_copy[['AQI','Humidity','CO','Dew','NO2','O3','PM10','PM2.5','SO2', 'Temperature','Wind']] = df_copy[['AQI','Humidity','CO','Dew','NO2','O3','PM10','PM2.5','SO2', 'Temperature','Wind']].replace(['-', 0], np.NaN).astype('float64')
df_1 = df_copy[['AQI','Humidity','CO','NO2','O3','PM10','PM2.5','SO2', 'Temperature','Wind']]
#thay the gia tri nan thanh so
df_1["AQI"].fillna(df_1["AQI"].median(), inplace = True)
df_1["Humidity"].fillna(df_1['Humidity'].mean(), inplace = True)
df_1["CO"].fillna(df_1['CO'].median(), inplace = True)
df_1["NO2"].fillna(df_1['NO2'].median(), inplace = True)
df_1["O3"].fillna(df_1['O3'].median(), inplace = True)
df_1["PM10"].fillna(df_1['PM10'].median(), inplace = True)
df_1["PM2.5"].fillna(df_1['PM2.5'].mean(), inplace = True)
df_1["SO2"].fillna(df_1['SO2'].median(), inplace = True)
df_1["Temperature"].fillna(df_1['Temperature'].mean(), inplace = True)
df_1["Wind"].fillna(df_1['Wind'].mean(), inplace = True)
#chay ket qua
# print(df_1.head())
#Xay dung model
X = df_1.drop("AQI", axis = 1)
y = df_1['AQI']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=.7, random_state = 0)
#chon random forest
model = RandomForestRegressor(n_estimators= 20)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#Qua trinh xay dung model ket thuc
#Qua trinh xuat thanh file plk
filename = 'air-quality-predict_1.pkl'
pickle.dump(model, open(filename, "wb"))