import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

train_len = int(len(data['Open']) * .9)
train = data[-train_len:]
test = data[:-(train_len - len(data))]

# Create forecast Cumulative Moving Average
y_hat_cma = data.copy()
y_hat_cma['sma_forecast'] = data['Open'].expanding().mean()
y_hat_cma['sma_forecast'][train_len:] = y_hat_cma['sma_forecast'][train_len-1]

# Test for accuracy

# Cumulative moving average accuracy test
mapec = mean_absolute_percentage_error(test['Open'], y_hat_cma['sma_forecast'][train_len:]).round(2)

results = pd.DataFrame({'Method': ['Cumulative moving average forecast'],
                        'MAPE': [mapec]})

results = results[['Method', 'MAPE']]
print(results)

# Show plots
plt.figure(figsize=(20, 5))
plt.grid()
plt.plot(test['Open'], label='Test', color='red')
plt.plot(y_hat_cma['sma_forecast'][:-(train_len - len(y_hat_cma['sma_forecast']))], label='Cumulative moving average '
                                                                                          'forecast', color='blue')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()
