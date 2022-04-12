import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


# Read data in from csv and organize properly
data = pd.read_csv('15-22_eth_prices.csv')
data.columns = ['Date', 'Open']
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d/%Y")
data = data.set_index('Date')

# first 2500 days as training set
train_len = 2500
train = data[-train_len:]
# last 192 days as out-of-time test set
test = data[:-(train_len - len(data))]

# Choose moving average window size
ma_window = 7

# Create forecast Simple Moving Average
y_hat_sma = data.copy()
y_hat_sma['sma_forecast'] = data['Open'].rolling(ma_window).mean()
y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1]

# Create forecast Exponential Moving Average
y_hat_ema = data.copy()
y_hat_ema['sma_forecast'] = data['Open'].rolling(ma_window).mean()
y_hat_ema['sma_forecast'][train_len:] = y_hat_ema['sma_forecast'][train_len-1]

# Create forecast Cumulative Moving Average
y_hat_cma = data.copy()
y_hat_cma['sma_forecast'] = data['Open'].expanding(ma_window).mean()
y_hat_cma['sma_forecast'][train_len:] = y_hat_cma['sma_forecast'][train_len-1]

# Test for accuracy
# Simple moving average accuracy test
rmses = np.sqrt(mean_squared_error(test['Open'], y_hat_sma['sma_forecast'][train_len:])).round(2)
mapes = mean_absolute_percentage_error(test['Open'], y_hat_sma['sma_forecast'][train_len:]).round(2)

# Exponential moving average accuracy test
rmsee = np.sqrt(mean_squared_error(test['Open'], y_hat_ema['sma_forecast'][train_len:])).round(2)
mapee = mean_absolute_percentage_error(test['Open'], y_hat_ema['sma_forecast'][train_len:]).round(2)

# Cumulative moving average accuracy test
rmsec = np.sqrt(mean_squared_error(test['Open'], y_hat_cma['sma_forecast'][train_len:])).round(2)
mapec = mean_absolute_percentage_error(test['Open'], y_hat_cma['sma_forecast'][train_len:]).round(2)

results = pd.DataFrame({'Method': ['Simple moving average forecast',
                                   'Exponential moving average forecast',
                                   'Cumulative moving average forecast'],
                        'MAPE': [mapes, mapee, mapec],
                        'RMSE': [rmses, rmsee, rmsec]})

results = results[['Method', 'RMSE', 'MAPE']]
print(results)

# Show plots
plt.figure(figsize=(20, 5))
plt.grid()
plt.plot(train['Open'], label='Train')
plt.plot(test['Open'], label='Test')
plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')
plt.plot(y_hat_ema['sma_forecast'], label='Exponential moving average forecast')
plt.plot(y_hat_cma['sma_forecast'], label='Cumulative moving average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()
