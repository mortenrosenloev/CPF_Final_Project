#
# XGBoost-Based Evaluation of a Multi-Timeframe Strategy 
# for EUR/USD Intraday Trading
# 
# CPF Algorithmic Trading Final Project
#
# Morten Rosenløv Jensen
# May 2025
#

# Third-party
import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('seaborn-v0_8')
import statsmodels.api as sm
import scipy.stats as scs
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
)
from xgboost import XGBClassifier

"""
Helper functions used in the study: 
XGBoost-Based Evaluation of a Multi-Timeframe Strategy 
for EUR/USD Intraday Trading.

Functions are organised chronologically in the order they
are used in the main notebook.
"""

#  *** HELPER FUNCTIONS FOR SECTION 2 ***
def plot_histogram(fd, norm):
    """
    Plots histogram over returns data and adds a standard normal 
    distribution curve.
    The plot is organised with subplots for aech 
    FinancialData instance.

    Parameters:
    - fd: dictionary of class instances from the FInancialData class.
    - norm: option to normalize data.
    
    Returns:
    - Histogram sub-plots.
    """
    
    n = len(fd)  # number of timeframes
    cols = 3  # sub-plot columns
    rows = (n + cols - 1) // cols  # sub-plot columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f'Data normalization: {norm}', fontsize=16)
    
    # Flatten axes array
    axes = axes.flatten()
    
    norm_dict = {    
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(
            output_distribution='normal', 
            random_state=100),
        'standard': StandardScaler(),
        'min-max': MinMaxScaler()
    }                  
    # Use standard_scaler if norm_type not in scaler_dict
    scaler = norm_dict.get(norm, StandardScaler())
        
    for i, tf in enumerate(fd):
        df = fd[tf].data.copy()
        df['r'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        if norm is not None:
            df['r_'] = scaler.fit_transform(df['r'].values.reshape(-1, 1))
        else:
            df['r_'] = df['r']
        ax = axes[i]
        ax.hist(df['r_'], bins=100, density=True, label='log-returns')
        x = np.linspace(df['r_'].min(), df['r_'].max(), 200)
        pdf = scs.norm.pdf(x, df['r_'].mean(), df['r_'].std())
        ax.plot(x, pdf, 'r', lw=2.0, label='Normal PDF')
        
        ax.set_title(f'Histogram of log-returns: {tf}')
        ax.legend()
        
    # Remove potential empty sub-plot windows
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_qq(fd, norm):
    """
    Plots Quantile-Quantile diagram over returns data.
    The plot is organised with subplots for aech FinancialData instance.

    Parameters:
    - fd: dictionary of class instances from the FInancialData class.
    - norm: option to normalize data.

    Returns:
    - QQ sub-plots.
    """
    
    n = len(fd)  # number of timeframes
    cols = 3  # sub-plot columns
    rows = (n + cols - 1) // cols  # sub-plot columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f'Data normalization: {norm}', fontsize=16)
    
    # Flatten axes array
    axes = axes.flatten()
    
    norm_dict = {    
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(
            output_distribution='normal', 
            random_state=100),
        'standard': StandardScaler(),
        'min-max': MinMaxScaler()
    }                  
    # Use standard_scaler if norm_type not in scaler_dict
    scaler = norm_dict.get(norm, StandardScaler())
        
    for i, tf in enumerate(fd):
        df = fd[tf].data.copy()
        df['r'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        if norm is not None:
            df['r_'] = scaler.fit_transform(df['r'].values.reshape(-1, 1))
        else:
            df['r_'] = df['r']
        sm.qqplot(df['r_'], line='s', ax=axes[i])
        axes[i].set_title(f'QQ-plot: {tf}')
    
    # Remove potential empty sub-plot windows
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#  *** HELPER FUNCTIONS FOR SECTION 3 ***
def create_model(n_estimators=None, max_depth=None, learning_rate=None,
                 subsample=None, colsample_bytree=None, gamma=None,
                 reg_alpha=None, reg_lambda=None,
                 early_stopping_rounds=None, 
                 eval_metric=None, random_state=None):
    """
    Instantiates an XGBClassifier with the specified hyperparameters.

    Parameters:
    - n_estimators (int, optional): Number of boosting rounds.
    - max_depth (int, optional): Maximum tree depth for 
        base learners.
    - learning_rate (float, optional): Step size shrinkage used to 
      prevent overfitting.
    - subsample (float, optional): Subsample ratio of the 
      training instances.
    - colsample_bytree (float, optional): Subsample ratio of columns 
      when constructing each tree.
    - gamma (float, optional): Minimum loss reduction required to 
      make a further partition.
    - reg_alpha (float, optional): L1 regularization term on weights.
    - reg_lambda (float, optional): L2 regularization term on weights.
    - early_stopping_rounds (int, optional): Activates early stopping.
      Validation metric needs to improve at least once in every 
      early_stopping_rounds round(s).
    - eval_metric (str or list of str, optional): Evaluation metric(s) 
      to be used for validation data.
    - random_state (int, optional): Random seed for reproducibility.

    Returns:
    - XGBClassifier: An instance of the XGBClassifier with the 
      provided settings.
    """
    
    params = {}
    
    if n_estimators is not None:
        params['n_estimators'] = n_estimators
    if max_depth is not None:
        params['max_depth'] = max_depth
    if learning_rate is not None:
        params['learning_rate'] = learning_rate
    if subsample is not None:
        params['subsample'] = subsample
    if colsample_bytree is not None:
        params['colsample_bytree'] = colsample_bytree
    if gamma is not None:
        params['gamma'] = gamma
    if reg_alpha is not None:
        params['reg_alpha'] = reg_alpha
    if reg_lambda is not None:
        params['reg_lambda'] = reg_lambda
    if early_stopping_rounds is not None:
        params['early_stopping_rounds'] = early_stopping_rounds
    if eval_metric is not None:
        params['eval_metric'] = eval_metric
    if random_state is not None:
        params['random_state'] = random_state

    return XGBClassifier(**params)


def plot_strategy_returns(
    model_res,
    timeframes,
    lw=1.0,
    alpha=0.5,
    include_baseline=True
):
    """
    Creates plots of strategy returns excl. and incl. trading cost.

    Parameters:
    - model_res (dict): dictionary on output from XGBTraining` class.
        keys = timeframe.
    - timeframes (dict): dictionary of timeframes used.
    - lw (float): line width.
    - alpha (float): transparancy.
    - include_baseline (bool, optional): 
        include buy-and-hold baseline.

    Returns:
    - Two sub-plots of returns with and without trading cost.
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    if include_baseline:
        baseline = model_res['timing'].test_results['r'].cumsum().apply(np.exp)
        ax[0].plot(baseline, label='Buy-and-hold', lw=lw, alpha=1.0)
        ax[1].plot(baseline, label='Buy-and-hold', lw=lw, alpha=1.0)

    # Left plot: without trading cost
    ax[0].set_title('Cumulative returns excl. trading cost')
    for tf in timeframes:
        strategy = model_res[tf].test_results['s'].cumsum().apply(np.exp)
        ax[0].plot(strategy, label=tf, lw=lw, alpha=alpha)
    ax[0].tick_params(axis='x', rotation=90)
    ax[0].legend()

    # Right plot: including trading cost
    ax[1].set_title('Cumulative returns incl. trading cost')
    for tf in timeframes:
        strategy_tc = model_res[tf].test_results['s_tc'].cumsum().apply(np.exp)
        ax[1].plot(strategy_tc, label=f'{tf}_incl. tc', lw=lw, alpha=alpha)
    ax[1].tick_params(axis='x', rotation=90)
    ax[1].legend()

    plt.tight_layout()
    return fig, ax
    

#  *** HELPER FUNCTIONS FOR SECTION 4 ***
def calculate_cagr(data):
    """
    Calculates the annualized Compound Annual Growth Rate (CAGR)
    based on a time-indexed series of log returns.

    Parameters:
    - data (pd.Series): Time-indexed log returns 
        (e.g. daily or intraday).
      The index must be datetime-like and sorted in ascending order.

    Returns:
    - float: The compound annual growth rate (CAGR) as a decimal.
    """
    
    # Annualized
    total_return = np.exp(data.sum())
    total_time = (data.index[-1] - 
                  data.index[0]).total_seconds() / (60 * 60 * 24 * 365)
    cagr = total_return ** (1 / total_time) - 1
    return cagr
    

def calculate_drawdown(data):
    """
    Calculates the drawdown curve from a time-indexed series of 
    log returns.

    Parameters:
    - data (pd.Series): Time-indexed log returns 
        (e.g. daily or intraday).
      The index must be datetime-like and sorted in ascending order.

    Returns:
    - pd.Series: The drawdown series, representing percentage 
        declines from the running maximum of the equity curve.
    """
    
    equity_curve = np.exp(data.cumsum())
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown


def calculate_sharpe(data, risk_free_rate=0):
    """
    Calculates the annualized Sharpe ratio from a time-indexed 
        series of log returns.

    Parameters:
    - data (pd.Series): Time-indexed log returns.
      The index must be datetime-like and sorted.
    - risk_free_rate (float, optional): Annualized risk-free 
        rate as a decimal (e.g. 0.01 for 1%). Defaults to 0.

    Returns:
    - float: Annualized Sharpe ratio.
    """
    
    bars = len(data)
    total_time = (data.index[-1] - 
                  data.index[0]).total_seconds() / (60 * 60 * 24 * 365)
    bars_per_year = bars / total_time
    rf_per_bar = risk_free_rate / bars_per_year  # risk_free_rate per bar
    mu, std = data.mean(), data.std()
    sharpe = ((mu - rf_per_bar) * bars_per_year) / (
        std * np.sqrt(bars_per_year))
    return sharpe


def calculate_trade_count(data):
    """
    Calculates the annualized number of long and short trades.

    Parameters:
    - data (pd.Series): Time-indexed position series with 
        values 1 (long), 
      -1 (short), or 0 (neutral). Must be sorted chronologically.

    Returns:
    - tuple of floats: (annualized long trades, 
        annualized short trades)
    """
    
    total_time = (data.index[-1] - 
                  data.index[0]).total_seconds() / (60 * 60 * 24 * 365)
    long_trades = ((data == 1) & (data.shift(1) != 1)).sum() / total_time
    short_trades = ((data == -1) & (data.shift(1) != -1)).sum() / total_time
    return long_trades, short_trades


def calculate_win_rate(df):
    """
    Calculates the annualized number of winning and losing trades,
    and computes the win rate.

    Parameters:
    - df (pd.DataFrame): Must contain:
        - 'pos' column: position series (1, -1, or 0)
        - 'r' column: log returns per bar
      The index must be datetime-like and sorted.

    Returns:
    - tuple:
        - wins (float): annualized number of winning trades
        - losses (float): annualized number of losing trades
        - win_rate (float): share of trades that are profitable 
            (NaN if no trades)
    """

    df = df.copy()
    df['pos_shift'] = df['pos'].shift(1).fillna(0)

    # Set trade-id for new trades that are long or short (not neutral)
    df['new_trade'] = (df['pos'] != df['pos_shift']) & (df['pos'] != 0)
    df['trade_id'] = df['new_trade'].cumsum()

    # Keep only active trades and skip neutral positions
    active_trades = df[df['pos'] != 0].copy()

    # Determine trade direction and trade returns
    grouped = active_trades.groupby('trade_id')
    direction = grouped['pos'].first()
    returns = grouped['r'].sum()
    trade_returns = direction * returns

    # Calculate win-rate
    total_time = (df.index[-1] - 
                  df.index[0]).total_seconds() / (60 * 60 * 24 * 365)
    wins = (trade_returns > 0).sum() / total_time
    losses = (trade_returns < 0).sum() / total_time
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else np.nan
    return wins, losses, win_rate


def calculate_metrics(df):
    """
    Computes key performance metrics for a trading strategy.

    Expects a DataFrame containing:
    - 's': cumulative log returns 
        (strategy without transaction costs)
    - 's_tc': cumulative log returns 
        (strategy with transaction costs)
    - 'pos': position series (1, -1, or 0)
    - 'r': log returns

    Parameters:
    - df (pd.DataFrame): Time-indexed data with required columns.

    Returns:
    - pd.DataFrame: Transposed DataFrame with named 
        performance metrics:
        - CAGR, Sharpe, trade stats, win rate, max drawdown
    """
    
    cagr = calculate_cagr(df['s'])
    cagr_tc = calculate_cagr(df['s_tc'])
    long, short = calculate_trade_count(df['pos'])
    wins, losses, win_rate = calculate_win_rate(df)
    sharpe= calculate_sharpe(df['s'])
    sharpe_tc = calculate_sharpe(df['s_tc'])
    max_dd = calculate_drawdown(df['s_tc']).min()
    
    metrics_dict = {
        'cagr': round(cagr, 4),
        'cagr_tc': round(cagr_tc, 4),
        'long_trades': round(long, 0),
        'short_trades': round(short, 0),
        'total_trades': round(long + short, 0),
        'wins': round(wins, 0),
        'losses': round(losses, 0),
        'win_rate': round(win_rate, 4),
        'sharpe': round(sharpe, 4),
        'sharpe_tc': round(sharpe_tc, 4),
        'max_dd': round(max_dd, 4)
    }
    metrics = pd.DataFrame([metrics_dict]).T
    return metrics


#  *** HELPER FUNCTIONS FOR SECTION 5 ***
def create_meta_labels(X, df, lookahead, tp_multiplier, sl_multiplier):
    """
    Generates meta labels for a classification model based on trade 
    entries, take-profit (TP), stop-loss (SL), and lookahead horizon.

    For each entry signal (in `df['pos']`), this function simulates 
    forward price movement to determine if TP or SL is hit first 
    within the lookahead window.

    Parameters:
    - X (pd.DataFrame): Feature matrix to be filtered based on 
        valid entries. Must share index with `df`.
    - df (pd.DataFrame): Must include the following columns:
        - 'pos': signal positions (1, -1, 0)
        - 'open', 'high', 'low', 'close': price columns
        - 'ATRr_14': ATR-based volatility estimate
    - lookahead (int): Number of bars to look ahead for TP/SL.
    - tp_multiplier (float): TP distance as multiple of ATR.
    - sl_multiplier (float): SL distance as multiple of ATR.

    Returns:
    - X (pd.DataFrame): Filtered feature matrix for active 
        trades only.
    - y (pd.Series): Binary labels (1 for TP, 0 for SL).
    - df (pd.DataFrame): Original dataframe enriched with TP, SL and 
        label info.
    """
    
    # Create dataframe with model signals, price data and ATR
    df = df.copy()

    # Flag when a new entry is made
    df['entry'] = (df['pos'] != 0) & (df['pos'] != df['pos'].shift(1))
    
    # set take-profit and stop-loss price levels
    df['tp'] = np.where(
        df['entry'], 
        df['open'] + df['pos'] * tp_multiplier * df['ATRr_14'],
        np.nan
    )
    df['sl'] = np.where(
        df['entry'], 
        df['open'] - df['pos'] * sl_multiplier * df['ATRr_14'],
        np.nan
    )

    # Instantiate lists for loop calculation
    entry = df['entry'].values
    pos = df['pos'].values
    open_price = df['open'].values
    close_price = df['close'].values
    high_price = df['high'].values
    low_price = df['low'].values
    take_profit = df['tp'].values
    stop_loss = df['sl'].values
    label = np.zeros(len(df))

    # Loop through each line and see if long/short entries are made
    # and if they first hit tp (label = 1), sl (label = -1) 
    # or lookforward (label = +/-1)
    for i in range(len(df)):
        if entry[i] and pos[i] == 1:  # long
                for j in range(lookahead):
                    if i + j >= len(df):
                        break
                    if low_price[i + j] <= stop_loss[i]:  # sl hit
                        label[i] = -1
                        break
                    if high_price[i + j] >= take_profit[i]:  # tp hit
                        label[i] = 1
                        break
                    if j == lookahead - 1:  # timeout
                        label[i] = 0
                        
        if entry[i] and pos[i] == -1:  # short
                for j in range(lookahead):
                    if i + j >= len(df):
                        break
                    if high_price[i + j] >= stop_loss[i]:  # sl hit
                        label[i] = -1
                        break
                    if low_price[i + j] <= take_profit[i]:  # tp hit
                        label[i] = 1
                        break
                    if j == lookahead - 1:  # timeout
                        label[i] = 0

    # Add labels to the dataframe and filter so that only datapoints
    # with active trades are maintained
    df['label'] = label
    df = df.loc[df['entry'] == True]

    # Reindex to only maintain datapoints with position entry
    X = X.reindex(df.index)  
    y = df['label'].replace(-1, 0)  # Binary labels only
    return X, y, df
    return df, trades_df
