import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def author():
    return "xsu73"
# 1. MACD
def macd(df):
    # 1. Get macd
    values = ema(df, 12) - ema(df, 26)
    # 2. Get signal
    signal = values.copy()
    signal.ix[35:, :] = ema(signal.ix[35:, :], 9)
    # signal = ema(values, 14)
    # 3. Discretize and get output
    # temp = values - signal
    temp = signal / values
    res = temp.copy()

    # res.ix[np.where(temp > 0.20)[0], :] = 2
    # res.ix[np.where(temp <= 0.20)[0], :] = -1
    # res.ix[np.where(temp <= 0)[0], :] = 0
    # res.ix[np.where(temp <= -0.4)[0], :] = 1
    # res.ix[temp.ix[:, 0].isnull(), :] = -2
    # res = res.astype(int)
    #
    # res.ix[:, :] = 0
    # signs = np.sign(temp).rolling(2).sum()
    # idx1 = np.where(signs == 0)[0]
    # idx2 = np.where(signs < 0)[0]
    # idx3 = np.where(signs > 0)[0]
    # buy = np.array([i for i in idx1 if i-1 in idx2])
    # sell = np.array([i for i in idx1 if i-1 in idx3])
    # res.ix[buy, :] = -1
    # res.ix[sell, :] = 1
    # res.ix[temp.ix[:, 0].isnull(), :] = -2
    # res = res.astype(int)
    return res


# 2. RSI
def rsi(df, lookback=14):
    # 1. Get rsi values
    dr = df.copy()
    dr.values[1:, :] = df.values[1:, :] - df.values[:-1, :]
    dr.values[0, :] = np.nan
    up_rets = dr[dr >= 0].fillna(0).cumsum()
    down_rets = -1 * dr[dr < 0].fillna(0).cumsum()

    up_gain = df.copy()
    up_gain.ix[:, :] = 0
    up_gain.values[lookback:, :] = up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]

    down_loss = df.copy()
    down_loss.ix[:, :] = 0
    down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]

    rs = (up_gain / lookback) / (down_loss / lookback)
    values = 100 - (100 / (1 + rs))
    values.ix[:lookback, :] = np.nan
    values[values == np.inf] = 100

    # 2. Get discretized index and output
    res = values.copy()
    # res.ix[np.where(values > 65)[0], :] = -1
    # idx1 = np.where(values >= 30)[0]
    # idx2 = np.where(values <= 65)[0]
    # idx = np.array([i for i in idx1 if i in idx2])
    # res.ix[idx, :] = 0
    # res.ix[np.where(values < 30)[0], :] = 1
    # res.ix[res.ix[:, 0].isnull(), :] = -2
    # res = res.astype(int)

    # res = values.copy()
    # res.ix[np.where(values > 72)[0], :] = 4
    # res.ix[np.where(values <= 72)[0], :] = 3
    # res.ix[np.where(values <= 63)[0], :] = 2
    # res.ix[np.where(values <= 55)[0], :] = 1
    # res.ix[np.where(values <= 40)[0], :] = 0
    # res.ix[np.where(values <= 32)[0], :] = -1
    # res.ix[res.ix[:, 0].isnull(), :] = -2

    # res = values.copy()
    # res.ix[:, :] = 0
    # signs = np.sign(values - 65).rolling(2).sum()
    # idx1 = np.where(signs == 0)[0]
    # idx3 = np.where(signs > 0)[0]
    # sell = np.array([i for i in idx1 if i-1 in idx3])
    #
    # signs = np.sign(values - 30).rolling(2).sum()
    # idx1 = np.where(signs == 0)[0]
    # idx2 = np.where(signs < 0)[0]
    # buy = np.array([i for i in idx1 if i - 1 in idx2])
    #
    # res.ix[idx2, :] = -1
    # res.ix[buy, :] = -1
    # res.ix[idx3, :] = 1
    # res.ix[sell, :] = 1
    # res.ix[values.ix[:, 0].isnull(), :] = -2
    # res = res.astype(int)
    return res


# 3. Stochastic indicator
def stochastic(df, df_high, df_low):
    # 1. Get stochastic values
    lookback = 20
    high = df_high.rolling(window=lookback).max()
    low = df_low.rolling(window=lookback).min()
    k = (df - low) / (high - low) * 100

    # 2. Get signal line
    d = k.rolling(3).mean()

    # 3. Get Inactive period indexes
    bottom_idx = np.where(k >= 15)[0]
    top_idx = np.where(k <= 88)[0]
    idx = np.array([i for i in bottom_idx if i in top_idx])

    # 4. Get discretized index and output
    values = d - k
    res = values.copy()

    res.ix[:, :] = 0

    idx1 = np.where(values < 0)[0]
    idx2 = np.where(k < 30)[0]
    neg_idx = np.array([i for i in idx1 if i in idx2])
    res.ix[neg_idx, :] = -1

    idx1 = np.where(values > 0)[0]
    idx2 = np.where(k > 70)[0]
    pos_idx = np.array([i for i in idx1 if i in idx2])
    res.ix[pos_idx, :] = 1
    res.ix[values.ix[:, 0].isnull(), :] = -2
    res = res.astype(int)


    return res


def ema(df, lookback=10):
    # lag = df.shift(period)
    res = df.copy()
    res.ix[:lookback - 1, :] = np.nan
    k = 2 / (1 + lookback)
    res.ix[lookback - 1, :] = df.ix[:lookback, :].mean(axis=0)
    for i in range(lookback, res.shape[0]):
        res.ix[i, :] = df.ix[i, :] * k + res.ix[i - 1, :] * (1 - k)
    return res


# 1. Bollinger bands
def bb(df, window=30):
    # define window size
    rm = df.rolling(window=window).mean()
    rm_std = df.rolling(window=window).std()
    upper_band = rm + rm_std * 2
    lower_band = rm - rm_std * 2
    values = (df - lower_band) / (upper_band - lower_band)

    res = values.copy()
    # res.ix[np.where(values > 1)[0], :] = -1
    #
    # idx1 = np.where(values >= 0.0)[0]
    # idx2 = np.where(values <= 1)[0]
    # idx = np.array([i for i in idx1 if i in idx2])
    # res.ix[idx, :] = 0
    #
    # res.ix[np.where(values < 0.0)[0], :] = 1
    #
    # res.ix[res.ix[:, 0].isnull(), :] = -2
    # res = res.astype(int)
    return res


# 4. Momentum
def momentum(df, lookback=30):
    temp = df / df.shift(lookback - 1) - 1
    res = temp.copy()

    res.ix[np.where(temp > 0.15)[0], :] = 2
    res.ix[np.where(temp <= 0.15)[0], :] = -1
    res.ix[np.where(temp <= 0.05)[0], :] = 0
    res.ix[np.where(temp <= -0.2)[0], :] = 1
    res.ix[temp.ix[:, 0].isnull(), :] = -2
    return res
