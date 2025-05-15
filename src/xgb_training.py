#
# XGBoost-Based Evaluation of a Multi-Timeframe Strategy for EUR/USD Intraday Trading
# CPF Algorithmic Trading Final Project
#
# Morten Rosenløv Jensen
# May 2025
#

import numpy as np
import pandas as pd
from pylab import plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

# Set seaborn style globally
sns.set()

class XGBTraining():
    """
    A class for training and evaluating an XGBoost classifier 
    on time-series data for intraday EUR/USD trading strategies.

    The class includes:
    - Optional hyperparameter optimization using cross-validation.
    - Feature selection based on feature importance.
    - Support for validation splits and sample weighting with 
        time decay.
    - Strategy-based return evaluation and performance visualization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test feature set.
    y_test : pd.Series
        Test labels.
    tc : float, default=0
        Transaction cost per trade (used in strategy evaluation).
    returns : pd.Series
        Series of log-returns indexed to match X.
    model : XGBClassifier
        Predefined XGBoost model. If None, model will be created 
        during hyperparameter tuning.
    validation_split : float, optional
        Proportion of training data to use for validation.
    top_n_features : int, optional
        Select top N most important features for training.
    optimize_hyperparameters : bool, default=False
        Whether to optimize hyperparameters using RandomizedSearchCV.
    scoring : str, default='f1'
        Scoring metric for hyperparameter optimization.
    time_decay : float, optional
        Rate of exponential time decay applied to sample weights 
        (in hours).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, prints progress and metric information.
    """


    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, 
                 tc=0, returns=None, model=None, validation_split=None, 
                 top_n_features=None,
                 optimize_hyperparameters=False, scoring='f1',
                 time_decay=None, seed=None, verbose=True):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.returns = returns
        self.tc = tc
        self.model = model
        self.val_split = validation_split
        self.top_n_features = top_n_features
        self.time_decay = time_decay
        self.seed = seed
        self.verbose = verbose

        # Extract data from the preprocessed data class object
        if self.val_split is not None:
            self._create_validation_data()
            self.eval_set = [(self.X_val, self.y_val)]
        else:
            self.eval_set = None
            

        # Set class weigths in case of imbalanced data
        class_weights = compute_sample_weight(
            class_weight='balanced',
            y=self.y_train
        )

        # Add time decay to the sample.weights
        if self.time_decay is not None:
            timestamps = self.X_train.index
            end_time = timestamps.max()
            age_in_hrs = (end_time - timestamps) / np.timedelta64(1, 'h')
            time_decay = np.exp(-self.time_decay * age_in_hrs)
            self.sample_weights = class_weights * time_decay
            
            # avoid too extreme difference:
            self.sample_weights = np.log(1 + self.sample_weights) 

        else:
            self.sample_weights = class_weights
        
        # Optional selection of top_n_features
        if self.top_n_features is not None:
            self._select_top_n_features()    
        
        # If optimization of hyperparameters, we create a model
        # based on optimized parameters
        if optimize_hyperparameters:
            self.scoring = scoring
            self._optimize_hyperparams()

        self._fit_perform()
        self._performance_metrics()
        self._strategy_results()

    
    def _create_validation_data(self):
        """
        Splits the training data into training and validation sets 
        based on the specified validation split ratio. Uses temporal 
        order (no shuffle) to preserve time series structure.
        """
        
        # Extract full training and test data sets
        self.X_train_full = self.X_train
        self.y_train_full = self.y_train
        
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X_train_full, self.y_train_full,
                             test_size=self.val_split,
                             random_state=self.seed,
                             shuffle=False
                            )

        
    def _select_top_n_features(self):
        """
        Trains the model to compute feature importances and selects 
        the top N features (based on gain) to reduce dimensionality 
        and improve model efficiency.
        """
        
        # Intermediate model training on the whole dataset 
        #to get feature importance
        self.model.fit(self.X_train, self.y_train,
                       sample_weight=self.sample_weights,
                       eval_set=self.eval_set, verbose=False)        

        importance_dict = self.model.get_booster().get_score(
            importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })

        # Sort and select top features
        importance_df = importance_df.sort_values(
            by='Importance', ascending=False)
        self.top_features = importance_df['Feature'].head(
            self.top_n_features).tolist()
        
        self.X_train = self.X_train[self.top_features]
        self.X_train_full = self.X_train_full[self.top_features]
        self.X_test = self.X_test[self.top_features]
        if self.val_split is not None:
            self.X_val = self.X_val[self.top_features]
            self.eval_set = [(self.X_val, self.y_val)]

    
    def _optimize_hyperparams(self):
        """
        Performs hyperparameter optimization using RandomizedSearchCV 
        with time-series cross-validation. 
        Updates the model with the best estimator.
        """
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.4, 0.5, 0.6, 0.7],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
            'reg_alpha': [0, 0.2, 0.5],
            'reg_lambda': [0, 1, 5]
        }
    
        base_model = XGBClassifier(early_stopping_rounds=10, 
                                   eval_metric='logloss',
                                   random_state=self.seed)
        
        tscv = TimeSeriesSplit(n_splits=4)
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=25,
            cv=tscv,
            scoring=self.scoring,
            verbose=1,
            n_jobs=-1,
            random_state=self.seed
        )
        
        search.fit(self.X_train, self.y_train,
                   sample_weight=self.sample_weights,
                   eval_set=self.eval_set, verbose=False)
        self.best_params = search.best_params_
        
        if self.verbose:
            print('Best parameters found:')
            print(self.best_params)

        self.model = search.best_estimator_

    
    def _fit_perform(self):
        """
        Fits the XGBoost model on the training data using optional
        validation set and sample weights. 
        Stores predictions for training, validation, and test sets.
        """
        
        self.model.fit(self.X_train, self.y_train,
                       sample_weight=self.sample_weights,
                       eval_set=self.eval_set, verbose=False)
        
        self.train_full_pred = self.model.predict(self.X_train_full)
        self.train_pred = self.model.predict(self.X_train)
        self.val_pred = self.model.predict(self.X_val)
        self.test_pred = self.model.predict(self.X_test)

    
    def _performance_metrics(self):
        """
        Calculates standard classification metrics 
        (accuracy, precision, recall, F1, ROC AUC)
        for train, validation, and test datasets. 
        Optionally prints the metrics.
        """        
        
        self.metrics = {
            'train': self._calculate_metrics(self.y_train, self.train_pred),
            'val': self._calculate_metrics(self.y_val, self.val_pred),
            'test': self._calculate_metrics(self.y_test, self.test_pred)
        }

        if self.verbose:
            print(86 * '-')
            print(f'TRAINING (IN-SAMPLE) METRICS:')
            for k, v in self.metrics['train'].items():
                print(f'{k}: {v:.4f}', end=' | ')
            print('\n' + 86 * '-')
            print(f'VALIDATION METRICS:')
            for k, v in self.metrics['val'].items():
                print(f'{k}: {v:.4f}', end=' | ')
            print('\n' + 86 * '-')
            print(f'TEST (OUT-OF-SAMPLE) METRICS:')
            for k, v in self.metrics['test'].items():
                print(f'{k}: {v:.4f}', end=' | ')
            print('\n' + 86 * '-')

    
    def _strategy_results(self):
        """
        Evaluates the financial performance of the strategy using 
        odel predictions, including return series and transaction 
        cost adjustments for both train and test sets.
        """        
        
        self.train_results = self._calculate_strategy_results(
            self.X_train_full, self.y_train_full, self.train_full_pred)
        self.test_results = self._calculate_strategy_results(
            self.X_test, self.y_test, self.test_pred) 

    
    # HELPER FUNCTIONS   
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculates performance metrics for given true and 
        predicted labels.

        Returns:
        -------
        dict
            Dictionary containing accuracy, precision, 
            recall, F1 score, and ROC AUC.
        """        
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred, average='weighted')
        }

    
    def _calculate_strategy_results(self, X, y_true, y_pred):
        """
        Calculates strategy returns based on directional predictions.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing true direction, predicted position, 
            raw returns, and net returns after transaction costs.
        """        
        
        
        df = pd.DataFrame(index=X.index)
        df['d'] = y_true
        df['pred'] = y_pred
        df['pos'] = np.where(df['pred'] > 0.5, 1, -1) # set long/short position
        df['r'] = self.returns.loc[X.index]
        df['s'] = df['pos'] * df['r']
        df['s_tc'] = np.where(df['pos'].diff() != 0, 
                              df['s'] - self.tc, df['s'])
        return df

    
    def plot_performance(self):     
        """
        Plots cumulative actual vs. predicted strategy returns for 
        both in-sample and out-of-sample periods. 
        Displays validation split if used.
        """        
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), 
                               constrained_layout=True)
        fig.suptitle(f'In-sample and out-of-sample performance', fontsize=14)
        # Left plot: in-sample performance
        train_perf = self.train_results[['r', 's']].cumsum().apply(np.exp)
        ax[0].plot(train_perf, label=['Actual', 'In-sample prediction'])
        # Add a vertical division line if validation data is used
        if self.val_split is not None:
            split_index = self.X_train.index[-1]
            ax[0].axvline(split_index, linestyle='--', color='gray', 
                          label='Validation split')
        
        ax[0].set_title('In-sample Performance')
        ax[0].legend(loc=2)
        ax[0].tick_params(axis='y', labelsize=8) 
        ax[0].tick_params(axis='x', rotation=90, labelsize=8) 
        
        # Right plot: out-of-sample performance
        test_perf = self.test_results[['r', 's']].cumsum().apply(np.exp) 
        ax[1].plot(test_perf, label=['Actual', 'Out-of-sample prediction'])
        ax[1].set_title('Out-of-sample Performance')
        ax[1].legend(loc=2)
        ax[1].tick_params(axis='y', labelsize=8) 
        ax[1].tick_params(axis='x', rotation=90, labelsize=8) 
        plt.show();

    
    def plot_confusion_matrix(self):
        """
        Displays a confusion matrix and classification report 
        for test predictions.
        """
        
        y_test = self.y_test
        y_pred = self.test_pred
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
   
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
                    annot_kws={'size': 8}, ax = ax[0])
        ax[0].set_xlabel('Predicted', fontsize=10)
        ax[0].set_ylabel('Actual', fontsize=10)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=8)
        ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=8)
        ax[0].set_title(f'Confusion Matrix', fontsize=12)
        ax[1].axis('off')
        ax[1].text(0, 0.5, report, fontsize=10, family='monospace', 
                   verticalalignment='center')
        plt.tight_layout()
        plt.show();        
