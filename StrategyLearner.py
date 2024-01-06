""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Xiaolu Su (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: xsu73 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903736688 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import random
import QLearner as ql
import indicators as it
import numpy as np
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
import util as ut
import marketsimcode as mc
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # constructor  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self,
                 verbose=False,
                 impact=0.0,
                 commission=0.0,
                 alpha=0.5,
                 gamma=0.3,
                 rar=0.5):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  		 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		  		 		  		  		    	 		 		   		 		  
        self.commission = commission
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
    def author(self):
        return "xsu73"

    def plot(self, df, name):
        font = {'title': 15,
                'axis': 13,
                'legend': 11,
                'ticks': 11}
        date_form = DateFormatter("%b %Y")
        plt.figure(figsize=(10, 6))
        plt.plot(self.norm_price, color='black', label="JPM", linewidth=0.5)
        plt.plot(df, color='blue', label=name, linewidth=0.5)
        sell_date_idx = self.dates[np.where(df == -1)[0]]
        buy_date_idx = self.dates[np.where(df == 1)[0]]
        plt.vlines(x=sell_date_idx, ymin=0, ymax=52, color="red", linestyles='--', linewidth=0.5, alpha=0.5,
                   label='Sell Signal')
        plt.vlines(x=buy_date_idx, ymin=0, ymax=52, color="green", linestyles='--', linewidth=0.5, alpha=0.5,
                   label='Buy Signal')
        plt.ylim((-2, 2))
        plt.grid(linestyle='--', linewidth=0.5)
        plt.savefig(f'{name}.png')
        plt.clf()
    def cleandata(self, df):
        clean_df = df.fillna(method='ffill')
        clean_df = clean_df.fillna(method='bfill')
        return clean_df
    def get_reward(self, i, h, h_prime):
        n = 14
        if i < n:
            return 0
        else:
            a = h_prime - h
            curr = (self.cash[i - 1]
                    + (-a) * 1000 * self.price.ix[i, 0] * (1 - np.sign(-a) * self.impact)
                    - np.sign(np.abs(-a)) * self.commission
                    + h_prime * 1000 * self.price.ix[i, 0])
            prev = self.cash[i - n - 1] + self.equity[i - n - 1]
            ret = curr / prev - 1
            return -ret

            # return ret
    def updatep(self, i, h, h_prime):

        a = h_prime - h
        self.orders[i] = a
        if i == 0:
            prevcash = self.sv
        else:
            prevcash = self.cash[i-1]
        if a == 0:
            self.cash[i] = prevcash
            # self.equity[i] = h_prime * self.price.ix[i, 0] * 1000
        else:
            self.cash[i] = (prevcash
                            + (-a) * 1000 * self.price.ix[i, 0] * (1 - np.sign(-a) * self.impact)
                            - np.sign(np.abs(-a)) * self.commission)
        self.equity[i] = h_prime * self.price.ix[i, 0] * 1000
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

    def discretize(self, df, name, verbose=False):
        if name == 'rsi':
            max = 60
            min = 35
        elif name == 'bb':
            max = 0.65
            min = 0.25

        disc = df.copy()
        disc.ix[:, :] = 0
        disc.ix[np.where(df.isnull() == True)[0], :] = -2
        disc.ix[np.where(df < min)[0], :] = -1
        disc.ix[np.where(df > max)[0], :] = 1
        if verbose:
            self.plot(disc, name)
        disc = disc.astype(int)
        return disc, max, min

    # this method should create a QLearner, and train it for trading
    def add_evidence(  		  	   		  		 		  		  		    	 		 		   		 		  
        self,  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
        sv=10000
    ):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		  		 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        # add your code to do learning here  	=====================================================

        # 1. Get data
        self.dates = pd.date_range(sd, ed)
        self.symbol = symbol
        self.sv = sv
        price = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
        high = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="High")
        low = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Low")
        close = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Close")

        self.inactive = np.where(price.isnull() == True)[0]
        # 2. Clean data
        price = self.cleandata(price)
        high = self.cleandata(high)
        low = self.cleandata(low)
        close = self.cleandata(close)
        self.price = price

        # 3. Get indicator values
        rsi = it.rsi(price)
        dk = it.stochastic(close, high, low)
        bb = it.bb(price)

        self.dret = self.classifyy()

        private = False                                 # adjust
        drsi, rsi_max, rsi_min = self.discretize(rsi, 'rsi', verbose=private)
        dbb, bb_max, bb_min = self.discretize(bb, 'bb', verbose=private)

        if private:
            # self.plot(self.dret, 'dailyreturn')
            print('rsi', rsi_max, rsi_min)
            print('bb', bb_max, bb_min)

        # 5. get states
        n_rsi = len(np.unique(drsi))
        n_k = len(np.unique(dk))
        n_bb = len(np.unique(dbb))

        # indicators = pd.concat([1000*(dbb+2), 100*(drsi+2), 10*(dmacd+2), 1*(dk+2)], axis=1)                # adjust
        indicators = pd.concat([100*(dbb+2), 10*(drsi+2), 1*(dk+2)], axis=1)                # adjust


        indicators['Total'] = indicators.sum(axis=1)
        # print(indicators)
        check = pd.concat([indicators, self.dret+2], axis=1)
        states = indicators[['Total']].copy()

        n = int(str(n_bb - 1) + str(n_rsi - 1) + str(n_k - 1)) + 1                        # adjust
        action_table = np.array([-1, 1, 0])
        # action_table = np.array([0, 1, -1])
        self.action_table = action_table

        # 4. Train in QLearner
        j = 0
        temp_trades = price.copy()
        temp_trades.ix[:, :] = 0
        self.trades = temp_trades.copy()
        # 1. initialize
        # actions: 0 = no action, 1 = buy, 2 = sell
        learner = ql.QLearner(num_states=n,
                              num_actions=3,
                              alpha=self.alpha,
                              gamma=self.gamma,
                              rar=self.rar,
                              radr=0.9,
                              dyna=0,
                              verbose=False)
        prev_ret = -1
        cum_ret = 0
        while np.abs(cum_ret - prev_ret) > 0.0001 and j < 50:
        # while j < 50:
            prev_ret = cum_ret
            # initialize portfolio
            m = price.shape[0]
            self.holds = np.zeros(m)
            self.orders = np.zeros(m)
            self.cash = np.zeros(m)
            self.equity = np.zeros(m)
            # a. current state
            h = 0
            hi = np.where(action_table == h)[0][0]
            i = 0
            self.holds[i] = h
            s = states.ix[i, 0]
            # s = int(str(hi) + str(states.ix[i, 0]))
            # b. action
            hpi = learner.querysetstate(s)
            h_prime = action_table[hpi]
            if 1 in self.inactive:
                h_prime = h
                hpi = hi
            self.holds[i] = h_prime
            self.updatep(0, h, h_prime)
            for i in range(1, price.shape[0]):
                # a. current state
                s_prime = states.ix[i, 0]
                r = self.get_reward(i, h, h_prime)
                # c. query for an action
                h = h_prime
                hpi = learner.query(s_prime, r)
                h_prime = action_table[hpi]
                if i in self.inactive:
                    h_prime = h
                    hpi = hi
                self.holds[i] = h_prime
                self.updatep(i, h, h_prime)
            temp_trades = pd.DataFrame({symbol: self.orders})
            self.learner = learner
            cum_ret = (self.cash[-1] + self.equity[-1]) / (self.cash[0] + self.equity[0]) - 1

            j += 1

        #  PYTHONPATH=../:. python grade_strategy_learner.py


    # this method should use the existing policy and test it against new data
    def testPolicy(  		  	   		  		 		  		  		    	 		 		   		 		  
        self,  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		  		 		  		  		    	 		 		   		 		  
    ):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  		 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        # 1. Get data
        self.dates = pd.date_range(sd, ed)
        self.symbol = symbol
        price = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
        high = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="High")
        low = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Low")
        close = ut.get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False, colname="Close")

        inactive = np.where(price.isnull() == True)[0]
        # 2. Clean data
        price = self.cleandata(price)
        high = self.cleandata(high)
        low = self.cleandata(low)
        close = self.cleandata(close)
        self.price = price
        self.dret = self.classifyy()

        # 3. Get indicator values
        rsi = it.rsi(price)
        dk = it.stochastic(close, high, low)
        bb = it.bb(price)

        # discretization
        private = False  # adjust
        drsi, rsi_max, rsi_min = self.discretize(rsi, 'rsi', verbose=private)
        dbb, bb_max, bb_min = self.discretize(bb, 'bb', verbose=private)

        if private:
            print('rsi', rsi_max, rsi_min)
            print('bb', bb_max, bb_min)

        indicators = pd.concat([100*(dbb+2), 10*(drsi+2), dk+2], axis=1)
        # indicators = pd.concat([100*(bb+2), 10*(rsi+2), k+2], axis=1)
        # indicators = pd.concat([100*(rsi+2), 10*(macd+2), k+2], axis=1)                 # adjust
        indicators['Total'] = indicators.sum(axis=1)
        states = indicators[['Total']].copy()
        action_table = self.action_table

        learner = self.learner
        orders_df = price.copy()
        orders_df.ix[:, :] = 0
        h = 0
        hi = 1
        for i in range(price.shape[0]):
            # a. current state
            s = states.ix[i, 0]
            # c. query for an action / next holding position
            hpi = learner.querysetstate(s)
            h_prime = action_table[hpi]
            if i in inactive:
                orders_df.ix[i, 0] = 0
                h_prime = h
                hpi = hi
            else:
                orders_df.ix[i, 0] = h_prime - h
            h = h_prime
            hi = hpi
        orders_df = orders_df * 1000
        return orders_df
  		  	   		  		 		  		  		    	 		 		   		 		  

if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		  	   		  		 		  		  		    	 		 		   		 		  
