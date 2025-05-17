#
# XGBoost-Based Evaluation of a Multi-Timeframe Strategy 
# for EUR/USD Intraday Trading
# 
# CPF Algorithmic Trading Final Project
#
# Morten Rosenløv Jensen
# May 2025
#

# Standard library
from math import pi

# Third-party
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_ta as ta
from pylab import plt
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
)

# Own modules
from financial_data import FinancialData

sns.set()


class DataPreProcessing(FinancialData):
    """
    Generic pipeline for preprocessing financial time series data 
    for ML/DL modeling.

    Inherits from FinancialData and adds:
    - Feature engineering (trend, momentum, volatility, calendar, etc.)
    - Train/test splitting
    - Feature normalization (scaler auto-selection or manual)
    - Label binning and market regime clustering
    - Lag feature generation
    - Target variable (y) generation

    After initialization, the following attributes are available:
    - self.X_train, self.X_test: Features
    - self.y_train, self.y_test: Binary target labels
    - self.features: List of all final feature names
    """

    def __init__(self, base_url, symbol='EUR_USD',
                 start='2020-01-01', end='2025-03-31', timeframe=None,
                 bins=5, clusters=8, lags=10, normalize=None,
                 tt_split=0.8):
        """
        Initializes the preprocessing pipeline.

        Parameters:
        - base_url (str): Data source URL.
        - symbol (str): Ticker symbol.
        - start (str): Start date of data.
        - end (str): End date of data.
        - timeframe (str): Time interval (e.g., 'H1', 'D1').
        - bins (int): Number of quantile bins.
        - clusters (int): Number of KMeans clusters.
        - lags (int): Number of lagged features to create.
        - normalize (str): Type of normalization to apply.
        - tt_split (float or str): Train/test split (ratio or date).
        """
        
        # Init base class and attributes
        super().__init__(base_url, symbol, timeframe)
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end) - pd.Timedelta(seconds=1)
        self.bins = bins
        self.clusters = clusters
        self.lags = lags
        self.norm = normalize
        self.tt_split = tt_split

        # Run pipeline steps
        self._add_features()
        self._create_splits()
        self._prepare_initial_datasets()

        if self.norm is not None:
            self._normalize()

        self._add_bins_regime()
        self._add_lags()
        self._finalize_targets()

    
    def _add_features(self):
        """
        Calls all modular feature engineering components 
        to enrich the dataset.
        """
        
        self._add_basic_returns()
        self._add_trend_features()
        self._add_rolling_features()
        self._add_momentum_features()
        self._add_volatility_features()
        self._add_slope_features()
        self._add_price_action_features()
        self._add_volume_features()
        self._add_cross_features()
        self._add_calendar_features()
        self._finalize_feature_selection()

    
    def _add_basic_returns(self):
        """
        Helper function to _add_features. 
        Adds log-returns and directional movee.
        """
        
        self.data['r'] = (
            np.log(self.data['close'] / self.data['close'].shift(1))
        )
        self.data['d'] = np.where(self.data['r'] > 0, 1, -1)
        self.data['rets'] = self.data['r']

    
    def _add_trend_features(self):
        """
        Helper function to _add_features. 
        Adds trend features: ADX and AROON from Pandas-TA.
        """
        
        self.data.ta.adx(append=True)
        self.data.ta.aroon(append=True)

    
    def _add_rolling_features(self):
        """
        Helper function to _add_features. 
        Adds rolling derivatives to returns and price data.
        """
        
        self.data['r_5ma'] = self.data['r'].rolling(window=5).mean()
        self.data['r_5std'] = self.data['r'].rolling(window=5).std()
        self.data['price_ma10'] = self.data['close'].rolling(window=10).mean()
        self.data['price_ma20'] = self.data['close'].rolling(window=20).mean()
        self.data['price_ma_ratio'] = (
            self.data['close'] / self.data['price_ma10']
        )

    
    def _add_momentum_features(self):
        """
        Helper function to _add_features.
        Adds momentum features: AO, RSI_9, RSI_14 and MACD 
        from Pandas-TA.
        """
        
        self.data.ta.ao(append=True)
        self.data.ta.rsi(length=9, append=True)
        self.data.ta.rsi(length=14, append=True)
        self.data.ta.macd(append=True)

    
    def _add_volatility_features(self):
        """
        Helper function to _add_features. 
        Adds volatility features: ATR, BB from Pandas-TA 
        and rolling derivatives.
        """
        
        self.data.ta.atr(append=True)
        self.data['atr_volatility'] = (
            self.data['ATRr_14'].rolling(window=20).std()
        )
        self.data.ta.bbands(append=True)
        self.data.drop(
            columns=[col for col in self.data.columns if 'BBP' in col],
            inplace=True)
        self.data['atr_spike'] = (
            self.data['ATRr_14'] / self.data['ATRr_14'].rolling(20).mean()
        )
        
    
    def _add_slope_features(self):
        """
        Helper function to _add_features. 
        Adds slope derivatives to RSI_14, AO, ADX, ATR, price MA's 
        and return MA's.
        """
        
        self.data['RSI_14_change'] = (
            self.data['RSI_14'] - self.data['RSI_14'].shift(1)
        )
        self.data['AO_change'] = (
            self.data['AO_5_34'] - self.data['AO_5_34'].shift(1)
        )
        self.data['ADX_change'] = (
            self.data['ADX_14'] - self.data['ADX_14'].shift(1)
        )

        self.data['ma10_slope'] = np.arctan(
            (self.data['price_ma10'].shift(1) -
             self.data['price_ma10'].shift(5)) / 4
        )
        self.data['ma20_slope'] = np.arctan(
            (self.data['price_ma20'].shift(1) - 
             self.data['price_ma20'].shift(5)) / 4
        )
        self.data['rsi14_slope'] = np.arctan(
            (self.data['RSI_14'].shift(1) - 
             self.data['RSI_14'].shift(5)) / 4
        )
        self.data['atr_slope'] = np.arctan(
            (self.data['ATRr_14'].shift(1) - 
             self.data['ATRr_14'].shift(5)) / 4
        )
        self.data['r_5ma_slope'] = np.arctan(
            (self.data['r_5ma'].shift(1) - 
             self.data['r_5ma'].shift(5)) / 4
        )

    
    def _add_price_action_features(self):
        """
        Helper function to _add_features. 
        Adds price-action features: co, hl, oc, ch, cl, ohlc4.
        """
        
        self.data['co'] = self.data['close'] / self.data['open'] - 1
        self.data['hl'] = self.data['high'] / self.data['low'] - 1
        self.data['oc'] = self.data['open'] / self.data['close'] - 1
        self.data['ch'] = self.data['close'] / self.data['high'] - 1
        self.data['cl'] = self.data['close'] / self.data['low'] - 1
        self.data['ohlc4'] = (self.data['open'] + self.data['high'] +
                              self.data['low'] + self.data['close']) / 4

    
    def _add_volume_features(self):
        """
        Helper function to _add_features. 
        Adds volume features: PVO, CMO from Pandas-TA.
        """
        
        self.data.ta.pvo(append=True)
        self.data.ta.cmo(append=True)

    
    def _add_cross_features(self):
        """
        Helper function to _add_features. 
        Adds cross-features from the features above.
        """
        
        self.data['momentum_volatility'] = (
            self.data['RSI_14'] * self.data['ATRr_14']
        )
        self.data['price_rsi'] = self.data['close'] / self.data['RSI_14']
        self.data['price_atr'] = self.data['close'] / self.data['ATRr_14']
        self.data['adx_rsi_interaction'] = (
            self.data['ADX_14'] * self.data['RSI_14']
        )
        self.data['range_atr'] = (
            (self.data['high'] - self.data['low']) / self.data['ATRr_14']
        )
        self.data['price_ma10_slope'] = (
            self.data['price_ma10'] * self.data['ma10_slope']
        )
        self.data['momentum_price_action'] = (
            self.data['RSI_14'] * self.data['co']
        )

    
    def _add_calendar_features(self):
        """
        Helper function to _add_features. 
        Adds calendar features: day and tod (time of day).
        """
        
        day = self.data.index.weekday
        tod = self.data.index.hour + self.data.index.minute / 60
        self.data['day_sin'] = np.sin(2 * np.pi * day / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * day / 7)
        self.data['tod_sin'] = np.sin(2 * np.pi * tod / 24)
        self.data['tod_cos'] = np.cos(2 * np.pi * tod / 24)

    
    def _finalize_feature_selection(self):
        """
        Helper function to _add_features. 
        Drops na, removes unnecessary columns and creates a list of 
        base_features.
        """
        
        self.data.dropna(inplace=True)
        self._adjust_date_range()
        exclude = ['open', 'high', 'low', 'close', 'volume', 'd', 'r']
        self._base_features = [
            x for x in self.data.columns if x not in exclude
        ]

    
    def _adjust_date_range(self):
        """
        Helper function to _finalize_feature_selection. 
        Adjusts the data according to start/end dates.
        """
        
        if self.start < self.data.index[self.lags]:
            print('-' * 80)
            print(
                'The selected start date is earlier than the available data.'
            )
            print(f'The start date is adjusted to: '
                  f'{self.data.index[self.lags]}')
            print('-' * 80)
            self.start = self.data.index[0]
        if self.end > self.data.index[-1]:
            print('-' * 80)
            print('The selected end date is later than the available data.')
            print(f'The end date is adjusted to: {self.data.index[-1]}')
            print('-' * 80)
            self.end = self.data.index[-1]
        start_index = self.data.index.get_indexer(
            [self.start], method='pad')[0]
        end_index = self.data.index.get_indexer(
            [self.end], method='pad')[0] + 1
        start_index = max(0, start_index - self.lags)
        self.data = self.data.iloc[start_index:end_index].copy()  

    
    def _create_splits(self):
        """
        Creates train/test splits based on tt_split parameter.
        Can handle float (percentage) or string (date).
        """     
        
        if isinstance(self.tt_split, float):
            split_date = self.start + self.tt_split * (self.end - self.start)
        else:
            split_date = pd.to_datetime(self.tt_split)

        split_date = self.data.index[self.data.index >= split_date].min()
        split_index = self.data.index.get_loc(split_date)
        self.train_index = self.data.index[:split_index]
        self.test_index = self.data.index[split_index:]

    
    def _prepare_initial_datasets(self):
        """
        Prepares base datasets used for model training/testing before
        normalization and lag creation.
        """
        
        self.features = self._base_features.copy()
        if self.bins > 0:
            self.features.append('bin')
        if self.clusters > 0:
            self.features.append('regime')

        self.X_train = self.data.loc[
        self.train_index, 
        self._base_features
        ].copy()
        self.X_test = self.data.loc[
        self.test_index,
        self._base_features
        ].copy()
        self.train_rets = self.data.loc[self.train_index, 'r'].copy()
        self.test_rets = self.data.loc[self.test_index, 'r'].copy()


    def _normalize(self):
        """
        Normalizes features using specified or 
        automatic scaler selection.
        """
        
        if self.norm == 'auto':
            self.scalers = self._automatic_scaler_selection()
            for feature, scaler in self.scalers.items():
                scaled_feature = scaler.fit_transform(
                    self.X_train[feature].values.reshape(-1, 1))
                self.X_train[feature] = pd.DataFrame(
                    scaled_feature, index=self.X_train.index)
                scaled_feature = scaler.transform(
                    self.X_test[feature].values.reshape(-1, 1))
                self.X_test[feature] = pd.DataFrame(
                    scaled_feature, index=self.X_test.index)

        else:
            norm_dict = {
                'robust': RobustScaler(),
                'quantile': QuantileTransformer(
                    output_distribution='normal',
                    random_state=100),
                'standard': StandardScaler(),
                'min-max': MinMaxScaler()
            }

            if self.norm not in norm_dict:
                print(
                    f'Warning: {self.norm} is not a valid scaler. '
                    f'StandardScaler will be used instead.'
                    )

            # Use standard_scaler if norm_type not in scaler_dict
            scaler = norm_dict.get(self.norm, StandardScaler())
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_train = pd.DataFrame(self.X_train,
                                        index=self.X_train.index,
                                        columns=self.X_train.columns
                                        )
            self.X_test = scaler.transform(self.X_test)
            self.X_test = pd.DataFrame(self.X_test,
                                       index=self.X_test.index,
                                       columns=self.X_test.columns
                                       )

    
    def _automatic_scaler_selection(self):
        """
        Helper function to _normalize.
        Automatically selects best scaler for each feature 
        based on skewness/kurtosis.
        Returns:
        dict: mapping feature names to fitted scalers.
        """    
        
        scalers = {}

        for feature in self.X_train.columns:
            # feature_data = self.X_train[feature]

            # Calculate statistics
            mean = self.X_train[feature].mean()
            std = self.X_train[feature].std()
            feature_skew = skew(self.X_train[feature])
            feature_kurtosis = kurtosis(self.X_train[feature])

            # Define the scaler choice criteria based on statistics
            # If data is skewed or has high kurtosis
            if np.abs(feature_skew) > 1 and np.abs(feature_kurtosis) > 3:
                # If the data is very skewed or has extreme outliers, 
                # use QuantileTransformer to adjust the distribution.
                scaler = QuantileTransformer(output_distribution='normal')
            
            # If data is skewed or has high kurtosis
            elif np.abs(feature_skew) > 1 or np.abs(feature_kurtosis) > 3:
                # If data is skewed or has high kurtosis but doesn't 
                # require full distribution change, use RobustScaler
                scaler = RobustScaler()
            
            # If standard deviation is very low, Min-Max might be better
            elif std < 1e-5:  
                scaler = MinMaxScaler()

            # Otherwise, StandardScaler is a safe choice
            else:  
                scaler = StandardScaler()

            # Store the selected scaler for each feature
            scalers[feature] = scaler
        return scalers    
        

    def _add_bins_regime(self):
        """
        Adds quantile binning and/or KMeans market regime labels 
        to feature set.
        """
        
        if self.bins > 0:
            quant = []
            step = 1 / self.bins
            for n in range(1, self.bins):
                q = n * step
                quant.append(self.train_rets.quantile(q))

            self.X_train['bin'] = np.digitize(
                self.train_rets, bins=quant, right=True) / self.bins 
            self.X_test['bin'] = np.digitize(
                self.test_rets, bins=quant, right=True) / self.bins  

        # KMEANS REGIME DETECTION
        if self.clusters > 0:
            regime_features = ['AO_5_34', 'RSI_14',
                               'ADX_14', 'price_ma_ratio', 'atr_spike']

            # 1. Select regime training data without NaNs
            self.regime_data = self.X_train[regime_features].copy()

            # 2. Fit kmeans
            kmeans = KMeans(n_clusters=self.clusters,
                            random_state=100).fit(self.regime_data)

            # 3. Predict on full data (fill missing values)
            regime_input_train = self.X_train[regime_features].copy()
            regime_input_test = self.X_test[regime_features].copy()
            # fill missing values robustly
            regime_input_train = regime_input_train.ffill().bfill()
            # fill missing values robustly
            regime_input_test = regime_input_test.ffill().bfill()
            regime_labels_train = kmeans.predict(regime_input_train)
            regime_labels_test = kmeans.predict(regime_input_test)

            # 4. Store regime labels and perform min-max normalization
            self.X_train['regime'] = pd.Series(
                regime_labels_train, index=self.X_train.index) / self.clusters
            self.X_test['regime'] = pd.Series(
                regime_labels_test, index=self.X_test.index) / self.clusters

    
    def _add_lags(self):
        """
        Adds lagged versions of all selected features 
        (except calendar features beyond lag=1).
        """
        
        self._lags_df = pd.concat([self.X_train, self.X_test], axis=0)
        cols = []
        no_lag_features = ['day_sin', 'day_cos', 'tod_sin',
                           'tod_cos']  # features where wo only use lag1
        for col in self.features:
            for lag in range(1, self.lags + 1):
                if col not in no_lag_features:
                    col_ = f'{col}_lag{lag}'
                    cols.append(col_)
                    self._lags_df[col_] = self._lags_df[col].shift(lag)
                elif lag == 1:
                    col_ = f'{col}_lag{lag}'
                    cols.append(col_)
                    self._lags_df[col_] = self._lags_df[col].shift(lag)

        # set the new features to the lagged data to avoid foresight
        self.lag_features = cols  

    
    def _finalize_targets(self):
        """
        Sets final target variables (y_train, y_test) as binary labels
        and aligns with lagged feature datasets.
        """
        
        self.X_train = self._lags_df.loc[
        self.train_index, self.lag_features
        ].dropna()
        self.X_test = self._lags_df.loc[self.test_index, self.lag_features]

        train_index = self.X_train.index
        test_index = self.X_test.index

        y_train = self.data.loc[train_index, 'd'].copy()
        y_test = self.data.loc[test_index, 'd'].copy()

        self.y_train = np.where(y_train == 1, 1, 0)
        self.y_test = np.where(y_test == 1, 1, 0)

    
    def plot_KMean(self):
        """
        Plots inertia (within-cluster sum of squares) across different
        numbers of KMeans clusters using the elbow method.
        """
        
        inertia = []
        for k in range(1, 20):
            kmeans = KMeans(n_clusters=k, random_state=100)
            kmeans.fit(self.regime_data)
            inertia.append(kmeans.inertia_)
        # Plot the Elbow method
        plt.plot(range(1, 20), inertia, marker='o')
        plt.title(
            'Elbow Method for KMean to determine optimal number of clusters'
        )
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.show()
