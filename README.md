# XGBoost-Based Evaluation of a Multi-Timeframe Strategy for EUR/USD Intraday Trading

## About
This project investigates a multi-timeframe trading strategy for the EUR/USD currency pair using machine learning. By combining predictions from three distinct timeframes—12h (trend), 4h (signal), and 1h (timing)—the strategy aims to reduce noise and improve the quality of trade entries and exits. Trades are executed only when all three models agree on the market direction, enhancing signal precision and reducing false positives.

Furthermore, the project explores meta-labeling with Lopez de Prado’s triple-barrier method to filter out noisy signals and optimize trade selection. The XGBoost classifier (XGBClassifier) is employed to build predictive models for each timeframe.

## Key Features
Multi-timeframe approach combining long-, medium-, and short-term signals

Use of XGBClassifier for directional prediction on EUR/USD intraday data

Meta-labeling via the triple-barrier method for trade validation

Focus on reducing trading costs and improving risk-adjusted returns
