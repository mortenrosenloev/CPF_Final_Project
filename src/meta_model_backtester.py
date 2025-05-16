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


class MetaModelBacktester:
    """
    A class to perform event-based backtesting for meta-model predictions 
    using TP/SL levels and a timeout.
    
    Parameters:
    -----------
    df (pd.DataFrame): Price and signal data.
    tp_multiplier (float): Multiplier for take-profit distance.
    sl_multiplier (float): Multiplier for stop-loss distance.
    lookahead (int): Maximum number of bars to hold a trade.
    tc (float, default=0): Transaction cost per trade (round-trip).
    """    
    
    def __init__(self, df, tp_multiplier, sl_multiplier, lookahead, tc=0):
        """
        Initializes the backtester and automatically runt the backtest.
        """
        
        self.df = df.copy()
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.lookahead = lookahead
        self.tc = tc
    
        # Proload arrays
        self._init_data()
    
        # Run event-based backtesting
        self._run()
    
        # Update final DataFrame and list of trades
        self._finalize()
    
    
    def _init_data(self):
        """
        Prepares internal data structures, arrays, 
        and containers for the backtest.
        """
        
        self.signal = self.df['signal'].values
        self.atr = self.df['ATRr_14'].values
        self.open_price = self.df['open'].values
        self.close_price = self.df['close'].values
        self.high_price = self.df['high'].values
        self.low_price = self.df['low'].values
        self.returns = self.df['r'].values
        self.pred = self.df['pred'].values
    
        # Instantiate positions, strategy returns, and comments lists
        self.positions = np.zeros(len(self.df))
        self.strategy_returns = np.zeros(len(self.df))
        self.strategy_returns_tc = np.zeros(len(self.df))
        self.comments = np.full(len(self.df), np.nan, dtype=object)
    
        # Containers       
        self.active_positions = []
        self.closed_positions = []
    
    
    def _run(self):
        """
        Core loop of the backtest. Iterates through the data, 
        handles entries and exits, and tracks performance.
        """
        
        for i in range(len(self.df)):
            pos_change = 0  # Entries or exit counter for each bar
    
            if self.pred[i] == 1:
                # Call helper function to define trade
                trade = self._enter_trade(i)
    
                # Update list of active positions
                self.active_positions.append(trade)
    
                # Adjust strategy returns for the given bar with entry spread 
                self.strategy_returns_tc[i] -= 0.5 * self.tc
                
                self.comments[i] = (
                    f"Id: {trade['id']} | entry {trade['direction']}"
                )
                pos_change += trade['direction']
    
            # Loop through all active positions and 
            # check if tp, sl, or timeout is hit.
            # We iterate over a copy because the list 
            # may be altered during the loop.
            for p in self.active_positions.copy():
    
                # Call helper function to check for exit conditions
                exit_trade, bar_pnl, bar_pnl_tc, reason, exit_price = \
                    self._check_exit_conditions(p, i)
    
                self.strategy_returns[i] += bar_pnl
                self.strategy_returns_tc[i] += bar_pnl_tc
    
                p['pnl'] += bar_pnl
                p['pnl_tc'] += bar_pnl_tc
                p['bars_held'] += 1
    
                if exit_trade:
                    pos_change -= p['direction']
                    p.update({
                        'exit_index': i,
                        'exit_price': exit_price,
                        'reason': reason
                    })
                    self.comments[i] = reason
                    self.active_positions.remove(p)
                    self.closed_positions.append(p)
    
            if i > 0:
                self.positions[i] = self.positions[i - 1] + pos_change
            else:
                self.positions[i] = pos_change
    
    
    def _finalize(self):
        """
        Finalizes the results after backtesting, 
        populates the result DataFrame and trades list.
        """
        
        self.df['pos'] = self.positions
        self.df['s'] = self.strategy_returns
        self.df['s_tc'] = self.strategy_returns_tc
        self.df['comment'] = self.comments
        self.results = self.df.copy()
        self.trades = pd.DataFrame(self.closed_positions)
    
    
    def _enter_trade(self, i):
        """
        Creates a new trade dictionary when an entry signal is triggered.
    
        Args:
            i (int): Index of the current bar.
    
        Returns:
            dict: A dictionary with trade metadata and state.
        """
        
        direction = self.signal[i]
        entry_price = self.open_price[i]
        atr = self.atr[i]
    
        if direction == 1:
            tp = entry_price + self.tp_multiplier * atr
            sl = entry_price - self.sl_multiplier * atr
        else:
            tp = entry_price - self.tp_multiplier * atr
            sl = entry_price + self.sl_multiplier * atr
    
        # Collect unique trade values
        trade = {
            'id': len(self.closed_positions) + len(self.active_positions) + 1,
            'direction': direction,
            'entry_price': entry_price,
            'entry_index': i,
            'tp': tp,
            'sl': sl,
            'pnl': 0,
            'pnl_tc': -0.5 * self.tc,
            'bars_held': 1
        }
        return trade
    
        
    def _check_exit_conditions(self, p, i):
        """
        Evaluates whether a trade should be closed due to TP, SL, or timeout.
    
        Args:
            p (dict): The active trade.
            i (int): Index of the current bar.
    
        Returns:
            tuple: exit_trade (bool), bar_pnl (float), 
              bar_pnl_tc (float), reason (str), exit_price (float).
        """        
        
        direction = p['direction']
        entry_price = p['entry_price']
        tp = p['tp']
        sl = p['sl']
        bars_held = p['bars_held']
        trade_id = p['id']
    
        exit_trade = False
        reason = ""
        exit_price = None
    
        if direction == 1:
            if self.low_price[i] <= sl:
                exit_price = sl
                if bars_held == 1:
                    bar_pnl = np.log(exit_price / entry_price)
                else:
                    bar_pnl = np.log(exit_price / self.close_price[i - 1])
                    
                exit_trade = True
                reason = f'Id: {trade_id} | long_sl'
            elif self.high_price[i] >= tp:
                exit_price = tp
                if bars_held == 1:
                    bar_pnl = np.log(exit_price / entry_price)
                else:
                    bar_pnl = np.log(exit_price / self.close_price[i - 1])
                exit_trade = True
                reason = f'Id: {trade_id} | long_tp'
        else:
            if self.high_price[i] >= sl:
                exit_price = sl
                if bars_held == 1:
                    bar_pnl = -np.log(exit_price / entry_price)
                else:
                    bar_pnl = -np.log(exit_price / self.close_price[i - 1])
                exit_trade = True
                reason = f'Id: {trade_id} | short_sl'
            elif self.low_price[i] <= tp:
                exit_price = tp
                if bars_held == 1:
                    bar_pnl = -np.log(exit_price / entry_price)
                else:
                    bar_pnl = -np.log(exit_price / self.close_price[i - 1])
                exit_trade = True
                reason = f'Id: {trade_id} | short_tp'
    
        if not exit_trade:
            if bars_held >= self.lookahead:
                exit_price = self.close_price[i]
                bar_pnl = direction * (
                    np.log(exit_price / self.close_price[i - 1])
                )
                reason = f'Id: {trade_id} | timeout'
                exit_trade = True
            else:
                if bars_held == 1:
                    bar_pnl = direction * (
                        np.log(self.close_price[i] / entry_price)
                        )
                else:
                    bar_pnl = direction * self.returns[i]
    
        bar_pnl_tc = bar_pnl - 0.5 * self.tc if exit_trade else bar_pnl
        return exit_trade, bar_pnl, bar_pnl_tc, reason, exit_price
    
    
    def plot_performance(self):
        """
        Plots cumulative returns of buy-and-hold vs strategy
        (with/without trade cost).
        """        
        
        plt.figure(figsize=(10, 4))
        plt.title('Strategy returns', fontsize=14)
        plt.plot(self.results['r'].cumsum().apply(np.exp), 
                 label='Buy-and-hold')
        plt.plot(self.results['s'].cumsum().apply(np.exp), 
                 label='Strategy excl. trading cost')
        plt.plot(self.results['s_tc'].cumsum().apply(np.exp), 
                 label='Strategy excl. trading cost')
        plt.legend()
        plt.show();
    
    
    def plot_trades_histogram(self):    
        """
        Plots a histogram of trade P&Ls including trade cost.
        """
        
        plt.figure(figsize=(10, 4))
        plt.hist(self.trades['pnl_tc'], bins=30, 
                 color='steelblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.title('Histogram of P&L incl. trade cost', fontsize=14)
        plt.xlabel('Log-Return')
        plt.ylabel('Number of trades')
        plt.grid(True)
        plt.show();
    
    
    def plot_box_plot(self):
        """
        Displays a box plot of P&L distributions with and without trade cost.
        """
        
        plt.figure(figsize=(10, 4))
        
        renamed = self.trades[['pnl', 'pnl_tc']].rename(columns={
            'pnl': 'P&L excl. trade cost',
            'pnl_tc': 'P&L incl. trade cost'
        })
        
        renamed.plot(kind='box', grid=True)
        
        plt.title('Box-plot of P&L with and without trade cost', fontsize=14)
        plt.ylabel('Log-Return')
        plt.show();
