import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


# data import

def get_data(stocks,start,end):
    stockData = pdr.get_data_yahoo(stocks,start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return meanReturns, covMatrix

stockList = ['AMZN', 'MSFT', 'AAPL']
# stocks = [stock + '.AX' for stock in stockList]


endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 300)

meanReturns,covMatrix = get_data(stockList,startDate,endDate)
# print(meanReturns, covMatrix)


weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# print(weights)

# Simulation
NUM_SIM = 100
TIME_FRAME = 1000

meanMtrx = np.full(shape=(TIME_FRAME,len(weights)),fill_value=meanReturns)
meanMtrx = meanMtrx.T

portfolio_sim = np.full(shape=(TIME_FRAME,NUM_SIM),fill_value=0.0) # structire is (days x sims) where each column stores all values for n days in that sim 

initial_portfolio = 10000

for sim in range(0,NUM_SIM):
    Z = np.random.normal(size=(TIME_FRAME,len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanMtrx + np.inner(L,Z)
    portfolio_sim[:,sim] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initial_portfolio


plt.plot(portfolio_sim)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')

plt.title('MC Sim')
plt.show(block=True)

# temp = pdr.DataReader("AAPL", start = startDate, end = endDate, data_source='yahoo')['Close']

# print(temp)

# data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
# print(data)