import sys
import os
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Additional imports for causal inference
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import the uploaded modules
from uk_stock_analyser_1 import UKStockAnalyzer
from advanced_analytics_1 import AdvancedAnalytics
from causal_inference_analysis_2 import CausalInferenceAnalyzer


def main():
    """
    Main execution function for analysis pipeline
    """
    print("🇬🇧 UK Stock Market Analysis: Identifying True Drivers of Price Movements")
    print("="*80)

    try:
        # Initialize analyzer
        analyzer = UKStockAnalyzer(
            tickers=['HSBA.L', 'BP.L', 'AZN.L', 'GSK.L', 'VOD.L'],
            start_date="2020-01-01"
        )

        # Execute complete analysis pipeline
        print("\n EXECUTING ANALYSIS PIPELINE...")

        # Step 1: Fetch stock data
        stock_data = analyzer.fetch_stock_data()
        print(f" Stock data fetched: {len(stock_data):,} records")

        # Step 2: Fetch macroeconomic data
        macro_data = analyzer.fetch_macroeconomic_data()

        # Step 3: Merge datasets
        merged_data = analyzer.merge_datasets()

        # Step 4: Calculate technical indicators
        final_data = analyzer.calculate_technical_indicators()

        # Step 5: Add Lag features
        lagged_data = analyzer.add_lag_features()

        # Step 6: Generate complete analysis
        analyzer.create_complete_analysis()

        # Step 7: Generate summary report
        analyzer.generate_summary_report()

        print("\n ANALYSIS COMPLETED SUCCESSFULLY!")
        
        #if analyzer or analyzer.final_df is  None:
        #    print("Error: Analyzer or final_df is None")
        #    return None
        print("\n RUNNING ADVANCED ANALYTICS...")

        # Apply advanced analytics
        advanced = AdvancedAnalytics()
        analyzer.final_df = advanced.calculate_regime_indicators(analyzer.final_df)
        analyzer.final_df = advanced.calculate_risk_metrics(analyzer.final_df)
        analyzer.final_df = advanced.sector_analysis(analyzer.final_df)

        print(" Advanced analytics completed!")
        print(f" Enhanced dataset now contains {len(analyzer.final_df.columns)} features")

        # Display sample of enhanced data
        print("\n SAMPLE OF ENHANCED DATASET:")
        sample_cols = ['Date', 'Ticker', 'Close', 'Daily_Return', 'RSI_14',
                      'Market_Regime', 'Sector', 'Target_Direction']
        available_cols = [col for col in sample_cols if col in analyzer.final_df.columns]
        print(analyzer.final_df[available_cols].tail(10).to_string(index=False))
        
        # Initialize causal_inf
        causal_inf = CausalInferenceAnalyzer(analyzer.final_df, target_variable = 'Target_Direction')

        # Step 1: Granger Causality Analysis
        granger_results = causal_inf.granger_causality_analysis(max_lag=10)

        # Step 2: Build Causal Graph
        causal_graph = causal_inf.build_causal_graph()

        # Step 3: DoWhy Analysis (if available)
        #if DOWHY_AVAILABLE:
        dowhy_results = causal_inf.dowhy_causal_analysis()

        # Step 4: Sensitivity Analysis
        sensitivity_results = causal_inf.sensitivity_analysis()

        # Step 5: Prepare modeling dataset
        modeling_df = causal_inf.prepare_modeling_dataset()

        # Step 6: Generate Report
        final_report = causal_inf.generate_causal_report()

        print(f"\n CAUSAL INFERENCE ANALYSIS COMPLETED!")
        print(f"   Identified {len(causal_inf.causal_features)} causally significant features")

        return analyzer,causal_inf

    except Exception as e:
        print(f"\n ERROR: {e}")
        print("Please ensure you have the required data files or adjust the data sources.")
        return None



if __name__ == "__main__":
    analyzer,causal_inf = main()
