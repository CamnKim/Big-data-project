import pandas
import matplotlib.pyplot as mpl

def parser(x):
    return pandas.to_datetime(x, unit='s')

# get data setup and plot
df = pandas.read_csv('data.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#df['timestamp'] = pandas.to_datetime(df['timestamp'], unit='s') # convert ms to datetime
df.info()

# plot data
df.plot(kind = 'line', x = 'timestamp', y = 'open')
#pandas.plotting.autocorrelation_plot(df['open'])


mpl.show()
