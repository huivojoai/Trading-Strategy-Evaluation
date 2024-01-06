# Trading-Strategy-Evaluation

DESCRIPTION - What is this package about

    This package can be used to train a Manual Strategy and a Strategy Learner that utilizes QLearner on indicators 
    to achieve a higher return than the Benchmark, which holds the 1000 shares through out the same period.

EXECUTION - How to use the package 

    1. QLearner.py
    The implementation of QLearner is built in this file. The QLeaner can be initialized with the following inputs:
        - num_states (int)
        - num_actions (int)
        - alpha (float): learning rate between 0.0 - 1.0
        - gamma (float): random action rate between 0.0 - 1.0
        - rar (float): random action decay rate between .0 - 1.0
        - dyna (int): number of dyna updates
    In this project, QLearner is imported and initialized by StrategyLearner and its attribute querysetstates(s) is utilized to retrieve the optimal
    or random action without updating the policy. During the training process, the strategy will call its query(s, r) to update the
    policy with the new state and reward. During testing, only querysetstates(s) should be called to retrieve outputs from the learner.

    2. ManualStrategy.py
    The Manual Strategy that develops human-defined trading rules is called by testproject.py and experiment1.py to generate charts and 
    portfolio statistics for comparisons described in the report. It is first imported and initiated in the testproject.py by calling ManualStrategy(), 
    then it can return an in-sample trading dataframe by calling trainPolicy(symbol, sd, ed, sv) and return an out-of-sample trading
    dataframe by calling testPolicy(symbol, sd, ed, sv). 

    3. StrategyLearner.py
    The Strategy Learner or Q-Learning based strategy is built in this file. It is called by experiment1.py and experiment2.py to conduct
    the experiments related to this strategy learner. It can first be imported and initiated by calling StrategyLearner(impact, commission, alpha, gamma,
    rar), where alpha, gamma, and rar will then be used for the QLearner embeded in the StrategyLearner. Then add_evidence(symbol, sd, ed, sv)
    can be called to train the learner with the in-sample data, and call testPolicy(symbol, sd, ed, sv) to retrieve an out-of-sample trades 
    dataframe. 

    4. indicators.py
    This file contains the functions that retrieve indicator values of 5 indicators: BB, RSI, MACD, Stochastic, and MOM, while only BB, RSI,
    and Stochastic were used in this project. They are called by both ManualStrategy and StrategyLearner to provide the features for the stock
    data. Most indicators take the adjusted close dataframe except for Stochastic, which takes the Close, High, and Low price as inputs. And Stochastic
    has already discretize the indicator values because of its dual discretization criteria. Once the file is imported, each function can be called and 
    return a dataframe with the same date indexes and column name as the price dataframe.

    5. experiment1.py
    This file conducts the experiment 1 that compares the performances among the Manual Strategy, Startegy Learner, and the Benchmark. It is
    imported and its run() can be called by testproject.py to generate the charts necessary for the report. 

    6. experiment2.py
    This file conducts the experiment 1 that compares the perforamnces of Startegy Learner with different impact values. It is
    imported and its run() can be called by testproject.py to generate the charts necessary for the report. 

    7. marketsimcode.py
    This file can be imported and its compute_portvals(df, start_val, commission, impact) can be called to generate a portfolio of the stock of
    the tradind dataset. It retrieves the stock price data by using the symbol on the column name of the dataframe and select the appropriate 
    date range to match the indexes of the trading dataframe. The trading dataframe should contain the actual number of shares, where 1000 
    represents BUYING 1000 shares while -1000 represents SELLING 1000 shares. The legal trading values can be -2000, -1000, 0, 1000, and 2000. 
    0 represents doing nothing and keeping the current position. The outputs of this function will only have three unique values: -1000, 0, 1000, 
    which mean SHORT, CASHOUT, and LONG. When unmuted, its test_code(portfolio) takes in the portfolio of the trades and print out a brief reoprt
    of the statistics related to the portfolio, including Sharpe Ratio, Cumulative Return, Daily Return, Standard Deviation along with the same 
    statistics for the markets to see the comparisons.

    8. testproject.py
    If there's no change needed to be made on the other files, this file is the one that imports and call all the necessary functions to generate all 
    information needed in the report. It contains the process of outputing the charts of in-sample and out-of-sample Benchmark Portfolio vs. Manual
    Portfolio, and it also runs the experiment files to conduct the experiments in single run. 

    9. p8_strategyEval_report.pdf
    This report tells the stories of the Maual Strategy and Strategy Learner and how they reacted to the indicator outputs and generates trades on JPM.
    The two experiments are also described in details for insights on the learner comparisons and the influence of impact.

    10. README.txt
    Current file that is used to provide descriptions and step-by-step instructions to access the information necessary and to run the files listed above. 
    
    Abbreviations:
    sd = start_date
    ed = end_date
    sv = start_val

INSTALLATION - How to install and setup code that builds model

      1. Store the files within the same directory with a folder named "images".

      2. Download the environment.yml file from this link below:
            
        https://lucylabs.gatech.edu/ml4t/fall2023/local-environment/
      
      2. Create a conda environment using environment.yml file with the code below. Run the following lines in order.

        conda env create --file environment.yml 
        conda activate ml4t 
	 
      3. Go to the link below and download the archive.zip file that containts the .gzip and .xlsx files.

         https://www.dropbox.com/s/m93z1aq6cgno89p/strategy_evaluation_2021Fall.zip?dl=1
       
	  4. Open terminal, navigate to the directory, and put in the following codes to run the testproject.py
        
        PYTHONPATH=../:. python testproject.py

        These files can also be individually run:
        
        experiment1.py and environment2.py

        PYTHONPATH=../:. python environment1.py
        PYTHONPATH=../:. python evnrionment2.py
        


