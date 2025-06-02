import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta
import warnings
import logging
import os
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/sheri/Downloads/final project/Causal_Inference/log/advanced_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """
    Additional analytical tools for deeper insights
    """

    def __init__(self):
        # Create plots directory
        self.plots_dir = 'C:/Users/sheri/Downloads/final project/Causal_Inference/output'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            logger.info(f"Created plots directory: {self.plots_dir}")

    @staticmethod
    def calculate_regime_indicators(df):
        """
        Identify market regimes (bull/bear/sideways markets)
        """
        logger.info("Starting market regime indicators calculation")

        for ticker in df['Ticker'].unique():
            logger.info(f"Calculating regime indicators for {ticker}")
            ticker_data = df[df['Ticker'] == ticker].copy()

            # Rolling 60-day trend
            ticker_data['Trend_60D'] = ticker_data['Close'].pct_change(periods=60)

            # Market regime classification
            # Bull market (>10% gain in 60 days)
            # Bear market (<-10% loss in 60 days)
            conditions = [
                ticker_data['Trend_60D'] > 0.1,
                ticker_data['Trend_60D'] < -0.1,
            ]
            choices = ['Bull', 'Bear']
            ticker_data['Market_Regime'] = np.select(conditions, choices, default='Sideways')

            df.loc[df['Ticker'] == ticker, 'Market_Regime'] = ticker_data['Market_Regime']
            logger.info(f"Regime indicators calculated for {ticker}")

        logger.info("Market regime indicators calculation completed")
        return df

    @staticmethod
    def calculate_risk_metrics(df):
        """
        Calculate comprehensive risk metrics
        """
        logger.info("Starting risk metrics calculation")

        for ticker in df['Ticker'].unique():
            logger.info(f"Calculating risk metrics for {ticker}")
            ticker_data = df[df['Ticker'] == ticker].copy()

            # Value at Risk (95% confidence)
            ticker_data['VaR_95'] = ticker_data['Daily_Return'].rolling(window=252).quantile(0.05)

            # Maximum Drawdown
            cumulative_returns = (1 + ticker_data['Daily_Return']).cumprod()
            running_max = cumulative_returns.expanding().max()
            ticker_data['Drawdown'] = (cumulative_returns - running_max) / running_max
            ticker_data['Max_Drawdown'] = ticker_data['Drawdown'].rolling(window=252).min()

            # Sharpe Ratio (assuming risk-free rate from BoE rate)
            if 'Rate' in ticker_data.columns:
                excess_returns = ticker_data['Daily_Return'] - (ticker_data['Rate'] / 100 / 252)
                ticker_data['Sharpe_Ratio'] = excess_returns.rolling(window=252).mean() / ticker_data['Daily_Return'].rolling(window=252).std() * np.sqrt(252)

            df.loc[df['Ticker'] == ticker, ['VaR_95', 'Drawdown', 'Max_Drawdown', 'Sharpe_Ratio']] = ticker_data[['VaR_95', 'Drawdown', 'Max_Drawdown', 'Sharpe_Ratio']].values
            logger.info(f"Risk metrics calculated for {ticker}")

        logger.info("Risk metrics calculation completed")
        return df

    @staticmethod
    def sector_analysis(df):
        """
        Add sector-specific analysis
        """
        logger.info("Starting sector analysis")

        # Sector mapping for FTSE 100 sample
        sector_map = {
            'HSBA.L': 'Banking',
            'BP.L': 'Energy',
            'AZN.L': 'Healthcare',
            'GSK.L': 'Healthcare',
            'VOD.L': 'Telecommunications'
        }

        df['Sector'] = df['Ticker'].map(sector_map)
        logger.info(f"Sector mapping applied: {sector_map}")

        # Calculate sector momentum
        sector_performance = df.groupby(['Date', 'Sector'])['Daily_Return'].mean().reset_index()
        sector_performance.columns = ['Date', 'Sector', 'Sector_Return']

        df = df.merge(sector_performance, on=['Date', 'Sector'], how='left')

        logger.info("Sector analysis completed")
        return df

''''
if __name__ == "__main__":
    logger.info("Starting advanced analytics execution")

    # Run the main analysis
    analyzer = main()

    # Optional: Run advanced analytics if main analysis succeeded
    if analyzer and analyzer.final_df is not None:
        print("\n RUNNING ADVANCED ANALYTICS...")
        logger.info("Running advanced analytics on main analysis results")

        # Apply advanced analytics
        advanced = AdvancedAnalytics()
        analyzer.final_df = advanced.calculate_regime_indicators(analyzer.final_df)
        analyzer.final_df = advanced.calculate_risk_metrics(analyzer.final_df)
        analyzer.final_df = advanced.sector_analysis(analyzer.final_df)

        print(" Advanced analytics completed!")
        print(f" Enhanced dataset now contains {len(analyzer.final_df.columns)} features")
        logger.info(f"Advanced analytics completed - Enhanced dataset contains {len(analyzer.final_df.columns)} features")

        # Display sample of enhanced data
        print("\n SAMPLE OF ENHANCED DATASET:")
        sample_cols = ['Date', 'Ticker', 'Close', 'Daily_Return', 'RSI_14',
                      'Market_Regime', 'Sector', 'Target_Direction']
        available_cols = [col for col in sample_cols if col in analyzer.final_df.columns]
        print(analyzer.final_df[available_cols].tail(10).to_string(index=False))
        logger.info("Sample of enhanced dataset displayed")
    else:
        logger.error("Main analysis failed or returned None - cannot run advanced analytics")
'''
