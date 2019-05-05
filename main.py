import requests
from bs4 import BeautifulSoup
import pandas as pd

web_raw = requests.get('https://finance.yahoo.com/quote/AAPL/history?p=AAPL').text

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
print(data_df.head())
