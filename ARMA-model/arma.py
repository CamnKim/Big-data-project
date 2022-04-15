import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# Read data in from csv and organize properly
data = pd.read_csv('15-22_eth_prices.csv')
data.columns = ['Date', 'Open']
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d/%Y")
data = data.set_index('Date')
data = data.iloc[::-1]                

# split into train and test | 0.929 = 2500
train_size = (int) (len(data) * 0.9)

train = list(data[0:train_size]['Open'])
test = list(data[train_size:]['Open'])

# rolling ARIMA
model_pred = []
num_test = len(test)

for i in range(num_test):
  model = ARIMA(train, order = (5,1,5))
  model_fit = model.fit()
  output = model_fit.forecast()
  yhat = np.float64(output[0]).item()
  model_pred.append(yhat)
  test_val = test[i]

  train.append(test_val)

  print(str(i) + ': pred: ' + str(yhat) + ', actual: ' + str(test_val))

print(model_fit.summary())

# calculate error
mape = np.mean(np.abs(np.array(model_pred) - np.array(test)) / np.abs(test))
print('MAPE: ' + str(mape))

# graph the data
plt.figure(figsize=(15,9))
plt.grid(True)

date_range = data[train_size:].index

plt.xlabel('Date')
plt.ylabel('Open Prices')
plt.title('Ethereum Price Prediction (ARIMA)')
plt.plot(date_range, model_pred, 'red', marker='o', linestyle='dashed', label='ETH predicted')
plt.plot(date_range, test, 'blue', label='ETH actual')
plt.legend()
plt.show()