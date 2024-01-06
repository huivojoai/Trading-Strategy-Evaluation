""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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
import numpy as np
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  

def compute_portvals(orders_df, start_val=1000000, commission=9.95, impact=0.005):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		  		 		  		  		    	 		 		   		 		  
    # code should work correctly with either input

    # TODO: Your code here
    orders_df = orders_df.sort_index()
    # 1.1 Get all symbols
    symbol = orders_df.columns[0]
    # 1.2 date indexes
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    dates = pd.date_range(start_date, end_date)
    # 1.3. initialize portfolio df and price df
    price_df = get_data([symbol], dates=dates, addSPY=False, colname="Adj Close")
    price_df = price_df.fillna(method='ffill')
    price_df = price_df.fillna(method='bfill')

    # 2. Update equity
    df = orders_df.copy()
    df['Equity'] = df.ix[:, symbol].cumsum()
    df['Equity'] = df['Equity'] * price_df.ix[:, symbol]
    # 3. Update Cash
    prevcash = start_val
    df['Cash'] = start_val
    for i in range(orders_df.shape[0]):
        a = orders_df.ix[i, 0]
        df.ix[i, 'Cash'] = (prevcash + (-a) * price_df.ix[i, 0]
                            * (1 - np.sign(-a) * impact)
                            - np.sign(np.abs(-a)) * commission)
        prevcash = df.ix[i, 'Cash']
        #     h = a

    # 5. Calculate Portfolio Value
    # print(df)
    portvals = price_df.copy()
    portvals[symbol] = df['Cash'] + df['Equity']

    # 6. Fill na
    portvals = portvals.fillna(start_val)
  		  	   		  		 		  		  		    	 		 		   		 		  
    return portvals
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def test_code(port_df):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		  		 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		  		 		  		  		    	 		 		   		 		  

    # Process orders
    # market ($SPX)
    start_date = port_df.index[0]
    end_date = port_df.index[-1]
    dates = pd.date_range(start_date, end_date)

    spyvals = get_data(['$SPX'], dates)
    spyvals = spyvals[['$SPX']]

    cum_ret_SPY = spyvals.iloc[-1] / spyvals.iloc[0] - 1
    daily_ret_SPY = (spyvals.iloc[1:] / spyvals.iloc[:-1].values) - 1
    avg_daily_ret_SPY = daily_ret_SPY.mean(axis=0)
    std_daily_ret_SPY = daily_ret_SPY.std(axis=0)
    sharpe_ratio_SPY = 252 ** 0.5 * avg_daily_ret_SPY / std_daily_ret_SPY

    # portfolio
    cum_ret = port_df.iloc[-1] / port_df.iloc[0] - 1
    daily_ret = (port_df.iloc[1:] / port_df.iloc[:-1].values) - 1
    avg_daily_ret = daily_ret.mean(axis=0)
    std_daily_ret = daily_ret.std(axis=0)
    sharpe_ratio = 252 ** 0.5 * avg_daily_ret / std_daily_ret

    # Compare portfolio against $SPX  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    # print(f"Final Portfolio Value: {port_df[-1]}")

def author():
    return 'xsu73'

if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
