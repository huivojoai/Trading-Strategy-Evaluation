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
import experiment1 as exp1
import experiment2 as exp2

def author():
    return "xsu73"
def plot1(df, benchmark, long1, short1):
    # Styles
    font = {'title': 15,
            'axis': 13,
            'legend': 11,
            'ticks': 11}

    plt.figure(figsize=(10, 6))
    plt.vlines(x=long1, ymin=0.9, ymax=1.25, color="blue", linestyles='--', linewidth=1.5, label='Long')
    plt.vlines(x=short1, ymin=0.9, ymax=1.25, color="black", linestyles='--', linewidth=1.5, label='Short')
    plt.plot(benchmark, color='green', label="Benchmark")
    plt.plot(df, color='red', label="Manual Strategy")

    plt.title("Performance of Benchmark vs. Manual Strategy on In-Sample JPM", fontsize=font['title'])
    plt.xticks(fontsize=font['ticks'])
    plt.yticks(fontsize=font['ticks'])
    plt.xlabel("Date", fontsize=font['axis'])
    plt.ylabel("Normalized Portfolio Value", fontsize=font['axis'])
    plt.ylim((0.75, 1.35))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=font['legend'])
    plt.savefig(f'images/Figure1.png')
    plt.clf()

def plot2(df, benchmark, long2, short2):
    # Styles
    font = {'title': 15,
            'axis': 13,
            'legend': 11,
            'ticks': 11}

    plt.figure(figsize=(10, 6))
    plt.vlines(x=long2, ymin=0.9, ymax=1.2, color="blue", linestyles='--', linewidth=1.5, alpha=0.8,
               label='Long')
    plt.vlines(x=short2, ymin=0.9, ymax=1.2, color="black", linestyles='--', linewidth=1.5, alpha=0.8,
               label='Short')
    plt.plot(benchmark, color='green', label="Benchmark")
    plt.plot(df, color='red', label="Manual Strategy")

    plt.title("Performance of Benchmark vs. Manual Strategy on Out-of-Sample JPM", fontsize=font['title'])
    plt.xticks(fontsize=font['ticks'])
    plt.yticks(fontsize=font['ticks'])
    plt.xlabel("Date", fontsize=font['axis'])
    plt.ylabel("Normalized Portfolio Value", fontsize=font['axis'])
    plt.ylim((0.8, 1.35))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=font['legend'])
    plt.savefig(f'images/Figure2.png')
    plt.clf()

def report(cr_opt, std_opt, avg_opt, cr_ben, std_ben, avg_ben):
    f = open("p6_results.txt", "w+")
    f.write("================================================= \n")
    f.write("Optimal Strategy Metrics \n")
    f.write("================================================= \n")
    f.write(f"Cumulative return = {cr_opt} \n")
    f.write(f"Stdev = {std_opt} \n")
    f.write(f"Mean = {avg_opt} \n")
    f.write("================================================= \n")
    f.write("Benchmark Metrics \n")
    f.write("================================================= \n")
    f.write(f"Cumulative return = {cr_ben} \n")
    f.write(f"Stdev = {std_ben} \n")
    f.write(f"Mean = {avg_ben} \n")
    f.close()

if __name__ == "__main__":
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000

    np.random.seed(903736688)
    # # 1. Get Manual Portfolio
    learner = ms.ManualStrategy()
    manual_train = learner.trainPolicy(symbol="JPM",
                                      sd=start_date,
                                      ed=end_date,
                                      sv=start_val)
    dates1 = pd.date_range(start_date, end_date)
    long1 = dates1[np.where(manual_train == 2000)[0]]
    short1 = dates1[np.where(manual_train == -2000)[0]]
    manual_test = learner.testPolicy(symbol="JPM",
                                       sd=dt.datetime(2010, 1, 1),
                                       ed=dt.datetime(2011, 12, 31),
                                       sv=start_val)
    dates2 = pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
    long2 = dates2[np.where(manual_test == 2000)[0]]
    short2 = dates2[np.where(manual_test == -2000)[0]]
    manual_train_port = mc.compute_portvals(manual_train,
                                            start_val=start_val,
                                            commission=9.95,
                                            impact=0.005)
    manual_test_port = mc.compute_portvals(manual_test,
                                            start_val=start_val,
                                            commission=9.95,
                                            impact=0.005)
    norm_manual_train_port = manual_train_port / manual_train_port.ix[0, 0]
    norm_manual_test_port = manual_test_port / manual_test_port.ix[0, 0]

    private = False
    if private:
        print('manual in')
        mc.test_code(manual_train_port)  # delete
        print('manual out')
        mc.test_code(manual_test_port)  # delete

    # benchmark
    benchmark_trades = manual_train.copy()
    benchmark_trades.ix[:, :] = 0
    benchmark_trades.ix[0, 0] = 1000
    benchmark_out_trades = manual_test.copy()
    benchmark_out_trades.ix[:, :] = 0
    benchmark_out_trades.ix[0, 0] = 1000
    benchmark_port = mc.compute_portvals(benchmark_trades, start_val=start_val, commission=9.95, impact=0.005)
    benchmark_out_port = mc.compute_portvals(benchmark_out_trades, start_val=start_val, commission=9.95, impact=0.005)
    norm_benchmark_port = benchmark_port / benchmark_port.ix[0, 0]
    norm_benchmark_out_port = benchmark_out_port / benchmark_out_port.ix[0, 0]

    if private:
        print('benchmark insample')
        mc.test_code(benchmark_port)                           # delete
        print('benchmark outsample')
        mc.test_code(benchmark_out_port)                           # delete


    plot1(norm_manual_train_port, norm_benchmark_port, long1, short1)
    dates2 = pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
    plot2(norm_manual_test_port, norm_benchmark_out_port, long2, short2)

    # experiment 1
    exp1.run()
    exp2.run()







# Report: only use symbol: JPM
# period: 2008/01/01 - 2009/12/31
# starting cash: $100,000

#     PYTHONPATH=../:. python testproject.py