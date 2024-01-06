import numpy as np
import pandas as pd
import datetime as dt
import indicators as it
from util import get_data, plot_data
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import ManualStrategy as ms
import marketsimcode as mc
import StrategyLearner as sl


def author():
    return "xsu73"
def plot(low, medium, high):
    # Styles
    font = {'title': 15,
            'axis': 13,
            'legend': 11,
            'ticks': 11}

    plt.figure(figsize=(10, 6))
    plt.plot(low, color='purple', label="impact=0.0005")
    plt.plot(medium, color='blue', label="impact=0.01")
    plt.plot(high, color='black', label='impact=0.05')
    plt.title("Performance of Strategy Learner by Impact on In-Sample JPM", fontsize=font['title'])
    plt.xticks(fontsize=font['ticks'])
    plt.yticks(fontsize=font['ticks'])
    plt.xlabel("Date", fontsize=font['axis'])
    plt.ylabel("Normalized Portfolio Value", fontsize=font['axis'])
    plt.ylim((0.0, 1.35))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=font['legend'])
    plt.savefig(f'images/Figure4.png')
    plt.clf()

def run():
    start_val = 100000
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    # ============================= Strategy Learner ==============================
    # 3. Get Learning Portfolio
    # for a in [0.1, 0.15, 0.2, 0.25, 0.3]:
    #     for g in [0.75, 0.8, 0.85, 0.9]:
    #         for r in [0.65, 0.7, 0.75, 0.8, 0.9]:
    # for i in range(20):
    a = 0.3
    g = 0.8
    r = 0.5
    low = sl.StrategyLearner(verbose=False,
                                 impact=0.0005,
                                 commission=0,
                                 alpha=a,
                                 gamma=g,
                                 rar=r)
    medium = sl.StrategyLearner(verbose=False,
                                  impact=0.01,
                                  commission=0,
                                  alpha=a,
                                  gamma=g,
                                  rar=r)
    high = sl.StrategyLearner(verbose=False,
                                  impact=0.05,
                                  commission=0,
                                  alpha=a,
                                  gamma=g,
                                  rar=r)

    low.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=start_val)
    medium.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=start_val)
    high.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=start_val)

    low_trades = low.testPolicy(symbol='JPM',
                                        sd=start_date,
                                        ed=end_date,
                                        sv=start_val)
    medium_trades = medium.testPolicy(symbol='JPM',
                                         sd=start_date,
                                         ed=end_date,
                                         sv=start_val)
    high_trades = high.testPolicy(symbol='JPM',
                                         sd=start_date,
                                         ed=end_date,
                                         sv=start_val)
    low_port = mc.compute_portvals(low_trades, start_val, commission=0, impact=0.0005)
    norm_low_port = low_port / low_port.ix[0, 0]
    medium_port = mc.compute_portvals(medium_trades, start_val, commission=0, impact=0.005)
    norm_medium_port = medium_port / medium_port.ix[0, 0]
    high_port = mc.compute_portvals(high_trades, start_val, commission=0, impact=0.05)
    norm_high_port = high_port / high_port.ix[0, 0]


    plot(norm_low_port, norm_medium_port, norm_high_port)
    private = False
    if private:
        print('norm_low_port')
        mc.test_code(norm_low_port)  # delete
        print('norm_medium_port')
        mc.test_code(norm_medium_port)  # delete
        print('norm_high_port')
        mc.test_code(norm_high_port)  # delete