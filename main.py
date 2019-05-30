import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math
# plt.style.use('fivethirtyeight')


def web_scrap_data(url):
    web_raw = requests.get(url).text

    web_soup = BeautifulSoup(web_raw, 'html.parser')

    web_tables = web_soup.find_all('table')
    web_trs = web_tables[0].find_all('tr')

    # Represents each row of the table
    cleaned_data = []
    # List to temporary hold each column index in a row
    # So they can be appended to a proper row when finished reading a row
    temp = []
    # Loop to go through every row in table
    # HTML only loads up to 102 even though there area 253 trading days in 1 year
    for row in range(1, len(web_trs)):
        temp = []  # Clear the temp row after each iteration
        web_tds = web_trs[row].find_all('td')

        # Because dividend is displayed as an entire row, we filter it out
        if len(web_tds) == 7:
            # should go from 0 to 6 (7 columns)
            for row2 in range(0, len(web_tds)):
                temp.append(web_tds[row2].text)
            cleaned_data.append(temp)

    data_df = pd.DataFrame(cleaned_data)
    data_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj. Close', 'Volume']

    data_df['Date'] = data_df['Date'].str.replace(',', '')
    data_df['Volume'] = data_df['Volume'].str.replace(',', '').astype(int)
    data_df['Date'] = pd.to_datetime(data_df.Date)

    data_df['Open'] = data_df['Open'].astype(float)
    data_df['High'] = data_df['High'].astype(float)
    data_df['Low'] = data_df['Low'].astype(float)
    data_df['Close'] = data_df['Close'].astype(float)
    data_df['Adj. Close'] = data_df['Adj. Close'].astype(float)
    return data_df


# aapl = web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
#                       'period1=1514793600&'
#                       'period2=1519804800&interval=1d&filter=history&frequency=1d')
# aapl = aapl.append(web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
#                                   'period1=1519891200&'
#                                   'period2=1527750000&interval=1d&filter=history&frequency=1d'), ignore_index=True)
# aapl = aapl.append(web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
#                                   'period1=1527836400&'
#                                   'period2=1535698800&interval=1d&filter=history&frequency=1d'), ignore_index=True)
# aapl = aapl.append(web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
#                                   'period1=1535785200&'
#                                   'period2=1546243200&interval=1d&filter=history&frequency=1d'), ignore_index=True)

# aapl = aapl.sort_values(by='Date').reset_index(drop=True)
# aapl.to_csv('418data.csv', index=False)

aapl_full = pd.read_csv('418data.csv', header=0, index_col='Date', parse_dates=True)
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

trace = go.Candlestick(x=aapl.index,
                       open=aapl.Open,
                       high=aapl.High,
                       low=aapl.Low,
                       close=aapl.Close)
data = [trace]
iplot(data, filename='simple_candlestick')

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

output = predict.values
output[0] = output[0] + aapl['Close'][-1]
for i in range(1, len(output)):
    output[i] = output[i] + output[i-1]
print(np.round(output, 4))


def flask_prediction(days, model=result):

# aapl['Close'].plot(grid=True)
# daily_close = aapl['Close']
# daily_pct_change = daily_close / daily_close.shift(1) - 1
# daily_pct_change.hist(bins=50)
# plt.show()
# print(daily_pct_change.describe())


