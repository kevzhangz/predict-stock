import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM

def predictStock(stock):
  #Kode Saham perusahaan yang akan diuji
  stocks = stock
  #Tanggal data yang akan diuji (tahun, bulan, tanggal)
  start = dt.datetime(2018,1,1)
  end = dt.datetime(2020,1,1)
  data = web.DataReader(stocks, 'yahoo', start, end)

  #Scaler untuk meminimalisir data yang diuji
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

  #Berapa data dari hari sebelumnya yang akan kita uji
  prediction_days = 60

  #Persiapan data uji
  X_train = []
  y_train = []

  for x in range(prediction_days, len(scaled_data)):
      X_train.append(scaled_data[x-prediction_days:x, 0]) 
      y_train.append(scaled_data[x,0])

  X_train, y_train = np.array(X_train), np.array(y_train)
  #Bentuk ulang dari data train agar dapat berjalan di neural net
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


  #Pembangunan model untuk graf
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(units=1)) 

  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(X_train, y_train, epochs=100, batch_size=32)

  test_start = dt.datetime(2021,1,1)
  test_end = dt.datetime.now()
  test_data = web.DataReader(stocks, 'yahoo', test_start, test_end)
  actual_prices = test_data['Close'].values

  total_dataset = pd.concat((data['Close'],test_data['Close']), axis=0)
  model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days: ].values
  model_inputs = model_inputs.reshape(-1,1)
  model_inputs = scaler.transform(model_inputs)

  X_test = []
  for x in range(prediction_days, len(model_inputs)):
      X_test.append(model_inputs[x-prediction_days:x, 0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

  predicted_price = model.predict(X_test)
  predicted_price = scaler.inverse_transform(predicted_price)

  a = plt.plot(actual_prices, color='black',label='Actual Share price')
  plt.plot(predicted_price, color='green',label='Predicted Share price')
  plt.title(f"{stocks} Share Price prediction")
  plt.xlabel('Time')
  plt.ylabel(f'{stocks} Share Price')
  plt.legend()

  dir_path = os.path.dirname(os.path.realpath(__file__))
  plt.savefig(fr"""{dir_path}\static\hasil.png""")


  real_data = [model_inputs[len(model_inputs) - prediction_days : len(model_inputs)+1, 0]]
  real_data =  np.array(real_data)
  real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
  prediction = model.predict(real_data)
  prediction = scaler.inverse_transform(prediction)
  return f"Tomorrow's {stocks} share price: {prediction}"