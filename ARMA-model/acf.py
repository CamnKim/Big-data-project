from tracemalloc import start
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Read data in from csv and organize properly
data = pd.read_csv('15-22_eth_prices.csv')
data.columns = ['Date', 'Open']
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d/%Y")
data = data.set_index('Date')

# get first difference to make data stationary
first_diff = data.diff()[1:]
# plot acf to determine order for MA model
plot_acf(first_diff)

plt.show()

# from the data we see that the first lag is signifigant