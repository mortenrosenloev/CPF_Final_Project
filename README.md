# XGBoost-Based Evaluation of a Multi-Timeframe Strategy for EUR/USD Intraday Trading  

## About  
This project investigates a multi-timeframe trading strategy for the EUR/USD currency pair using machine learning. By combining predictions from three distinct timeframes—12h (trend), 4h (signal), and 1h (timing)—the strategy aims to reduce noise and improve the quality of trade entries and exits. Trades are executed only when all three models agree on the market direction, enhancing signal precision and reducing false positives.

Furthermore, the project explores meta-labeling with Lopez de Prado’s triple-barrier method to filter out noisy signals and optimize trade selection. The XGBoost classifier (`XGBClassifier`) is employed to build predictive models for each timeframe.

## Key Features  
- Multi-timeframe approach combining long-, medium-, and short-term signals
- Use of `XGBClassifier` for directional prediction on EUR/USD intraday data
- Meta-labeling via the triple-barrier method for trade validation
- Focus on reducing trading costs and improving risk-adjusted returns

## Introduction  
Analyzing market movements across multiple timeframes is a common practice among swing and day traders. Typically, the ratio between the individual timeframes ranges from 1:3 to 1:5. There are various ways to apply multi-timeframe analysis, including the use of indicators, support/resistance levels, and other technical methods.

In this study, we investigate the use of AI-based models (`XGBClassifier`) to predict the directional movement of the EUR/USD currency pair across three timeframes, with the goal of identifying high-quality entry and exit points for long and short positions. To the best of our knowledge, this specific use of AI models in a multi-timeframe framework has not been previously documented in the literature.

The three timeframes used in this study are:
- **12h – 'trend'**: This prediction determines the overall market direction over the next 12 hours and defines the permissible trading direction (long or short).
- **4h – 'signal'**: Trades are only permitted when the 4h prediction agrees with the 'trend'. If a trade is active, a change in the 4h signal also serves as an exit condition.
- **1h – 'timing'**: This prediction is used for precise entry timing. A trade is entered only when all three models — 'trend', 'signal', and 'timing' — align in the same direction.

The predictions from these three timeframes are combined into a Multi-Timeframe Strategy (MTS), capable of taking long or short positions, or remaining neutral. As a final layer, we apply the triple-barrier meta-labeling technique proposed by Lopez de Prado to further filter and validate trading signals.

## Project Structure  
CPF_Final_Project/  
   │  
   ├── README.md # This file  
   ├── requirements.txt # Python dependencies  
   ├── data/ # Data files (CSV)  
   │ ├── EUR_USD_2020-01-01_2025-03-31_M15_A.csv  # 15min ask-prices from OANDA  
   │ ├── EUR_USD_2020-01-01_2025-03-31_M15_B.csv  # 15min bid-prices from OANDA  
   │ ├── EUR_USD_2020-01-01_2025-03-31_M15_M.csv  # 15min mid-prices from OANDA  
   ├── src/ # Source code (modules)  
   │ ├── helper_functions.py  
   │ ├── financial_data.py  
   │ ├── data_pre_processing.py  
   │ └── xgb_training.py  
   └── CPF_Final_Project.ipynb # Main Jupyter Notebook with project code and commentary  

## Usage  
The project is primarily run from the Jupyter Notebook 'CPF_Final_Project.ipynb', which contains all code, explanations, and results. The notebook is designed to be executable on Google Colab without modifications.  

### Running in Google Colab
⚠️ Important note for Colab users:
Due to compatibility issues with the `pandas_ta` library, the required `numpy` version is downgraded in the first code block. This triggers a forced Colab kernel restart, which is intentional and necessary.
After the restart, simply re-run all cells from the top to continue execution without issues.

### To run locally:
1. Clone this repo
   bash
   git clone https://github.com/mortenrosenloev/CPF_Final_Project.git
   
2. Install dependencies (pip install -r requirements.txt)
3. Open 'CPF_Final_Project.ipynb' in Jupyter or VSCode
4. Run all cells in order

*Created by Morten Rosenløv Jensen*
   
