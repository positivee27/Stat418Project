import requests
from bs4 import BeautifulSoup
import pandas as pd


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


aapl = web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
                      'period1=1514793600&'
                      'period2=1519804800&interval=1d&filter=history&frequency=1d')
aapl = aapl.append(web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
                                  'period1=1519891200&'
                                  'period2=1527750000&interval=1d&filter=history&frequency=1d'), ignore_index=True)
aapl = aapl.append(web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
                                  'period1=1527836400&'
                                  'period2=1535698800&interval=1d&filter=history&frequency=1d'), ignore_index=True)
aapl = aapl.append(web_scrap_data('https://finance.yahoo.com/quote/AAPL/history?'
                                  'period1=1535785200&'
                                  'period2=1546243200&interval=1d&filter=history&frequency=1d'), ignore_index=True)

aapl = aapl.sort_values(by='Date').reset_index(drop=True)
aapl.to_csv('418data.csv', index=False)