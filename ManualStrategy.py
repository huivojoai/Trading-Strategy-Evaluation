import numpy as np
import os
import indicators as it
import datetime as dt
from util import get_data, plot_data
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')
class ManualStrategy(object):
    def __int__(self, verbose=False):
        self.verbose = verbose

    def cleandata(self, df):
        clean_df = df.fillna(method='ffill')
        clean_df = clean_df.fillna(method='bfill')
        return clean_df
    def discretize(self, df, name, verbose=False):
        if name == 'rsi':
            max = 60
            min = 35

        elif name == 'bb':
            max = 0.65
            min = 0.25
        # max = round(df.ix[np.where(self.dret == 1)[0], :].min() * long, 2)
        # min = round(df.ix[np.where(self.dret == -1)[0], :].max() * short, 2)

        disc = df.copy()
        disc.ix[:, :] = 0
        disc.ix[np.where(df.isnull() == True)[0], :] = -2
        disc.ix[np.where(df < min)[0], :] = -1
        disc.ix[np.where(df > max)[0], :] = 1
        if verbose:
            self.plot(disc, name)
        disc = disc.astype(int)
        return disc, max, min

    def plot(self, df, name):
        font = {'title': 15,
                'axis': 13,
                'legend': 11,
                'ticks': 11}
        date_form = DateFormatter("%b %Y")
        plt.figure(figsize=(10, 6))
        plt.plot(self.price, color='black', label="JPM", linewidth=0.5)
        sell_date_idx = self.dates[np.where(df < 0)[0]]
        buy_date_idx = self.dates[np.where(df > 0)[0]]
        plt.vlines(x=sell_date_idx, ymin=0, ymax=52, color="red", linestyles='--', linewidth=0.5, alpha=0.5,
                   label='Sell Signal')
        plt.vlines(x=buy_date_idx, ymin=0, ymax=52, color="green", linestyles='--', linewidth=0.5, alpha=0.5,
                   label='Buy Signal')
        plt.ylim((10, 55))
        plt.grid(linestyle='--', linewidth=0.5)
        plt.savefig(f'{name}1.png')
        plt.clf()

    def author(self):
        return "xsu73"  # replace tb34 with your Georgia Tech username

    def classifyy(self):
        b = 0.075
        s = -0.075
        self.norm_price = self.price / self.price.ix[0, 0]
        N = 14
        dret = (self.price / self.price.shift(N)) - 1
        line = dret.copy()
        line.ix[:, :] = 0
        line.ix[np.where(dret > b)[0], :] = 1
        line.ix[np.where(dret < s)[0], :] = -1
        return line
    def trainPolicy(
                    self,
                    symbol="IBM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000
                    ):
        self.dates = pd.date_range(sd, ed)
        self.symbol = symbol
        price = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
        high = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="High")
        low = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Low")
        close = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Close")

        self.inactive = np.where(price.isnull() == True)[0]

        # 2. Clean data
        price = self.cleandata(price)
        self.price = price
        high = self.cleandata(high)
        low = self.cleandata(low)
        close = self.cleandata(close)

        # 3. Get indicator values
        rsi = it.rsi(price)
        k = it.stochastic(close, high, low)
        bb = it.bb(price)

        self.private = False
        drsi, rsi_max, rsi_min = self.discretize(rsi, 'rsi', verbose=self.private)
        dbb, bb_max, bb_min = self.discretize(bb, 'bb', verbose=self.private)
        dk = k.copy()
        # 4. Get trade df
        train_df= self.get_trades(dbb, drsi, dk)
        return train_df


    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        # 1. Get data
        self.dates = pd.date_range(sd, ed)
        self.symbol = symbol
        price = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
        high = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="High")
        low = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Low")
        close = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Close")

        # 2. Clean data
        price = self.cleandata(price)
        self.price = price
        high = self.cleandata(high)
        low = self.cleandata(low)
        close = self.cleandata(close)

        # 3. Get indicator values
        # macd = it.macd(price)
        rsi = it.rsi(price)
        k = it.stochastic(close, high, low)
        bb = it.bb(price)

        drsi, rsi_max, rsi_min = self.discretize(rsi, 'rsi', verbose=self.private)
        dbb, bb_max, bb_min = self.discretize(bb, 'bb', verbose=self.private)
        dk = k.copy()

        # 4. Get trade df
        df = self.get_trades(dbb, drsi, dk)
        return df

    def get_trades(self, bb, rsi, k):

        # 2. short: bb + rsi + k
        short_signals = pd.concat([rsi, k, bb], axis=1)
        votes = short_signals.median(axis=1)
        short_num_idx = np.where(votes == -1)[0]
        long_num_idx = np.where(votes == 1)[0]

        df = self.price.copy()
        df.ix[:, :] = 0
        df.ix[long_num_idx, :] = 1
        df.ix[short_num_idx, :] = -1
        # self.plot(df, 'try')
        h = 0
        # trades = df.copy()
        for i in range(df.shape[0]):
            hprime = df.ix[i, 0]
            if hprime == 0 and h != 0:
                hprime = h
                df.ix[i, 0] = h
            h = hprime

        trades = df - df.shift(1)
        trades.ix[0, 0] = 0
        trades = trades * -1000
        return trades


        #   PYTHONPATH=../:. python experiment1.py