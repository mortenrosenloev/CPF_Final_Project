#
# XGBoost-Based Evaluation of a Multi-Timeframe Strategy for EUR/USD Intraday Trading
# CPF Algorithmic Trading Final Project
#
# Morten Rosenløv Jensen
# May 2025
#

# Third-party
import pandas as pd
import numpy as np
import pytz
from datetime import timedelta
import statsmodels.api as sm
import scipy.stats as scs
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler, 
    QuantileTransformer
)

class FinancialData():    
    """ 
    Handles import and processing of OHLC price data and volume 
    for FX-pairs or indices.

    Data are timestamped in the UK timezone but follow US daylight 
    saving time (DST). 
    All resampling starts from the first data point at the beginning 
    of each week regardless of DST changes. 
    Resampled data use `label='right'` to avoid foresight 
    — timestamps are moved from the 'open' to the 'close' 
    of the new interval.

    Attributes
    ----------
    url : str
        URL base path to CSV files.
    symbol : str
        Financial instrument symbol (e.g., 'EUR_USD').
    timeframe : str or None
        Pandas-compatible resample frequency string (e.g., '1D', '1h').

    Methods
    -------
    _retrieve_data()
        Loads raw OHLC data and drops unnecessary columns.
    _resample_data()
        Applies DST-based logic and resamples data to desired timeframe.
    _calculate_trading_cost()
        Loads bid/ask prices and calculates average spread and 
        log trading cost.
    """
        
    
    def __init__(self, url, symbol='EUR_USD', timeframe=None):
        """
        Initiates the class and executes private methods.

        Parameters:
        - url (str): Data source URL.
        - symbol (str): Ticker symbol.
        - timeframe (str): Time interval (e.g., 'H1', 'D1').
        """
        
        self.url = url
        self.symbol = symbol
        self.tf = timeframe
        self.data = pd.DataFrame()
        self._retrieve_data()
        self._resample_data()
        self._calculate_trading_cost()
        
    
    def _retrieve_data(self):
        """
        Loads raw price data from CSV and 
        removes non-essential columns.
        """
        
        suffix = f'_2020-01-01_2025-03-31_M15_M'
        filename = f'{self.symbol}{suffix}.csv'
        file = self.url + filename
        self.raw = pd.read_csv(file, index_col=0, parse_dates=True)
        del self.raw['complete']

    
    def _resample_data(self):
        """
        Resamples the data according to the specified timeframe, 
        accounting for US daylight saving time (DST). 
        Applies different UTC offsets depending on whether a 
        timestamp falls under DST.
        """
        
        if self.tf:
            # Create a timezone-aware index for DST adjustment
            dummy_index = (
                self.raw.index.
                tz_localize('Europe/London').
                tz_convert('America/New_York')
            )
            self.raw['us_dst'] = dummy_index.map(lambda x: x.dst() != timedelta(0))
            
            # Split data into Summer time and Winter time sets
            dst_true_df = self.raw.loc[self.raw['us_dst'] == True].copy()
            dst_false_df = self.raw.loc[self.raw['us_dst'] == False].copy()
            
            resample_dict = {
                'o': 'first',
                'h': 'max',
                'l': 'min',
                'c': 'last',
                'volume': 'sum'
            }
        
            # Resample using different UTC offsets
            res_1 = dst_true_df.resample(self.tf, label='right', closed='left', 
                                         offset='21h').agg(resample_dict)
            res_2 = dst_false_df.resample(self.tf, label='right', closed='left', 
                                          offset='22h').agg(resample_dict)
            self.data = pd.concat([res_1, res_2]).dropna().sort_index()
        
        else: # no resampling needed - return a copy of raw
            self.data = self.raw.copy()
        
        self.data.columns = ['open', 'high', 'low', 'close', 'volume']
        self.data.dropna(inplace=True)

    
    def _calculate_trading_cost(self):
        """
        Imports ask and bid price data and calculates the 
        average spread and mean log-return of the trading cost 
        due to the bid/ask spread.
        """
        
        raw = {}
        for price in ['A', 'B']:
            suffix = f'_2020-01-01_2025-03-31_M15_{price}'
            filename = f'{self.symbol}{suffix}.csv'
            file = self.url + filename
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            raw[price] = df['c'].copy()
        self.spread = (raw['A'] - raw['B']).mean()
        self.tc = np.log(raw['A'] / raw['B']).mean()