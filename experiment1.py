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
def plot(manual, benchmark, learner):
    # Styles
    font = {'title': 15,
            'axis': 13,
            'legend': 11,
            'ticks': 11}

    plt.figure(figsize=(10, 6))
    plt.plot(benchmark, color='green', label="Benchmark")
    plt.plot(manual, color='red', label="Manual Strategy")
    plt.plot(learner, color='blue', label='Strategy Learner')
    plt.title("Performance of Benchmark vs. Manual Strategy vs. Strategy Learner on In-Sample JPM", fontsize=font['title'])
    plt.xticks(fontsize=font['ticks'])
    plt.yticks(fontsize=font['ticks'])
    plt.xlabel("Date", fontsize=font['axis'])
    plt.ylabel("Normalized Portfolio Value", fontsize=font['axis'])
    plt.ylim((0.75, 1.35))
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=font['legend'])
    plt.savefig(f'images/Figure3.png')
    plt.clf()

def run():
    start_val = 100000
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)


    # ============================= Manual Strategy ==============================
    # Part 1: In-sample Benchmark vs. Manual
    # compare_port(norm_manual_port, norm_benchmark_port)

    # # 1. Get Manual Portfolio
    learner = ms.ManualStrategy()
    manual_train = learner.trainPolicy(symbol="JPM",
                                       sd=start_date,
                                       ed=end_date,
                                       sv=start_val)
    manual_train_port = mc.compute_portvals(manual_train,
                                            start_val=start_val,
                                            commission=9.95,
                                            impact=0.005)
    norm_manual_port = manual_train_port / manual_train_port.ix[0, 0]

    # benchmark
    benchmark_trades = manual_train.copy()
    benchmark_trades.ix[:, :] = 0
    benchmark_trades.ix[0, 0] = 1000
    benchmark_port = mc.compute_portvals(benchmark_trades, start_val=start_val, commission=9.95, impact=0.005)
    norm_benchmark_port = benchmark_port / benchmark_port.ix[0, 0]

    # ============================= Strategy Learner ==============================
    # 3. Get Learning Portfolio
    # for a in [0.1, 0.15, 0.2, 0.25, 0.3]:
    #     for g in [0.75, 0.8, 0.85, 0.9]:
    #         for r in [0.65, 0.7, 0.75, 0.8, 0.9]:
    # for i in range(20):
    a = 0.3
    g = 0.5
    r = 0.8
    learner = sl.StrategyLearner(verbose=False,
                                 impact=0.005,
                                 commission=9.95,
                                 alpha=a,
                                 gamma=g,
                                 rar=r)

    learner.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=start_val)
    learner_trades = learner.testPolicy(symbol='JPM',
                                        sd=start_date,
                                        ed=end_date,
                                        sv=start_val)
    learner_port = mc.compute_portvals(learner_trades, start_val, commission=9.95, impact=0.005)
    norm_learner_port = learner_port / learner_port.ix[0, 0]
    # print(f'alpha={a}, gamma={g}, rar={r}:')
    # mc.test_code(learner_port)  # delete
    #   PYTHONPATH=../:. python experiment1.py
    # dates = pd.date_range(start_date, end_date)
    plot(norm_manual_port, norm_benchmark_port, norm_learner_port)

    private = False
    if private:
        print('strategy learner')
        mc.test_code(norm_learner_port)  # delete