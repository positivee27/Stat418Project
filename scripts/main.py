import pandas as pd
import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# from plotly.offline import iplot
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima_process import ArmaProcess
# from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math
# plt.style.use('fivethirtyeight')


aapl_full = pd.read_csv('.\\scripts\\418data.csv', header=0, index_col='Date', parse_dates=True)
print(aapl_full.shape)
print(aapl_full.head())

### Split the data into training and test
train_size = len(aapl_full) - 10
aapl = aapl_full[0:train_size]
aapl_test = aapl_full[train_size-1:len(aapl_full)]


# Plot the Data
aapl['2018':'2018'].plot(subplots=True, figsize=(10, 12))
plt.title('Apple stock attributes in 2018')
plt.savefig('stocks.png')

# trace = go.Candlestick(x=aapl.index,
#                        open=aapl.Open,
#                        high=aapl.High,
#                        low=aapl.Low,
#                        close=aapl.Close)
# data = [trace]
# iplot(data, filename='simple_candlestick')

## Exploratory Analysis

### Decomposition Plots
rcParams['figure.figsize'] = 11, 9
decomposed_aapl_volume = sm.tsa.seasonal_decompose(aapl['Close'], freq=60)
figure = decomposed_aapl_volume.plot()
plt.title('Decomposition Plots')
plt.savefig('Decomposition Plots.png')

adf = adfuller(aapl['Close'])
print('p-value: {}' .format(float(adf[1])))
# p-value of 0.76 > 0.05 indicates non-stationarity in the data so we need to
# adjust the data to become stationnary

plot_acf(aapl['Close'], lags=25, title='AAPL Close ACF')
plt.title('Autocorrelation of Closing Price')
plt.savefig('ACF.png')
plot_pacf(aapl['Close'], lags=25, title='AAPL Close PACF')
plt.title('Partial Autocorrelation of Closing Price')
plt.savefig('PACF.png')
# Plotting the ACF and PACF we see that there is large autocorrelation within the lagged values, and we
# see geometric decay in our plots. This indicates we will have to transform our data to be stationary

aapl['Close_diff'] = aapl['Close'].diff()
aapl.iloc[0, 6] = 0
temp = pd.DataFrame(aapl['Close_diff'])
temp.reset_index(level=0, inplace=True)
temp['Close_diff'].plot(figsize=(20, 6))
plt.title('Differenced Closing Price')
plt.savefig('Differenced Closing Price.png')

adf = adfuller(aapl['Close_diff'])
print('p-value: {}' .format(float(adf[1])))

plot_acf(aapl['Close_diff'], lags=25, title='AAPL Differenced Close ACF')
plt.title('Autocorrelation of Differenced Closing Price')
plt.savefig('Differenced ACF.png')

plot_pacf(aapl['Close_diff'], lags=25, title='AAPl Differenced Close PACF')
plt.title('Partial Autocorrelation of Differenced Closing Price')
plt.savefig('Differenced PACF.png')

## Fitting ARMA Model

model_arma = ARIMA(aapl['Close_diff'], order=(0, 1, 1))
result = model_arma.fit()
print(result.summary())
result.plot_predict(start=200, end=250)
predict = result.predict(start=241, end=250)
print(predict)
plt.title('ARIMA Model Prediction')
plt.savefig('ARIMA Model.png')

aapl_test_diff = aapl_test['Close'].diff().dropna().values
rmse = math.sqrt(mean_squared_error(aapl_test_diff, predict))
print(rmse)


def forecast(days, model=result):
    y_pred = model.predict(start=241, end=240+days)
    output = y_pred.values
    output[0] = output[0] + aapl['Close'][-1]
    for i in range(1, len(output)):
        output[i] = output[i] + output[i - 1]
    # print(np.round(output, 4))
    return output


# print("predicting 1: ")
# print(forecast(10))
# print("preidcting 2: ")
# print(forecast(5))

# aapl['Close'].plot(grid=True)
# daily_close = aapl['Close']
# daily_pct_change = daily_close / daily_close.shift(1) - 1
# daily_pct_change.hist(bins=50)
# plt.show()
# print(daily_pct_change.describe())


