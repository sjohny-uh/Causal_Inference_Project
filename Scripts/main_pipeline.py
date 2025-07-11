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
from data_preprocessing import UKStockAnalyzer,run_data_preprocessing
from causal_inference import CausalInferenceAnalyzer, run_causal_analysis
from feature_selection import FeatureSelector,run_feature_selection
from stock_market_prediction import run_stock_prediction_analysis


def main():
    """
    Main execution function for enhanced analysis pipeline
    """
    print("UK Stock Market Analysis: Identifying True Drivers of Price Movements")
    print("="*80)

    try:
        # Initialize analyzer
        '''
        analyzer = UKStockAnalyzer(
            tickers=['HSBA.L','BP.L', 'AZN.L', 'GSK.L', 'VOD.L'],
            start_date="2015-01-01",
            end_date="2025-06-01"
        )
        '''
        analyzer = run_data_preprocessing()

        print("\nBASIC ANALYSIS COMPLETED SUCCESSFULLY!")

        print(f"Dataset now contains {len(analyzer.final_df.columns)} features")

        # Display sample of  data
        print("\nSAMPLE OF DATASET:")
        sample_cols = ['Date', 'Ticker', 'Close', 'Daily_Return', 'RSI_14',
                      'Market_Regime', 'Sector', 'Target_Direction']
        available_cols = [col for col in sample_cols if col in analyzer.final_df.columns]
        print(analyzer.final_df[available_cols].tail(10).to_string(index=False))

        # Run complete causal inference analysis
        print("\nRUNNING CAUSAL INFERENCE ANALYSIS...")
        print("   This includes:")
        print("   - Granger Causality Analysis")


        causal_inf, causal_results = run_causal_analysis(
            df=analyzer.final_df,
            target_variable='Target_Direction'
        )

        print(f"\nCAUSAL INFERENCE ANALYSIS COMPLETED!")
        print(f"   Identified {len(causal_inf.causal_features)} causally significant features")


        print("\nRUNNING FEATURE SELECTION ANALYSIS...")
        selector, metadata = run_feature_selection(df=causal_inf.modeling_df, method='random_forest')

        modeling_df = selector.modeling_df

        # Step 7: Run Prediction Analysis
        print("\nRUNNING PREDICTION ANALYSIS...")
        print("   Using causally-informed features for improved predictions...")

        try:
            pipeline, results = run_stock_prediction_analysis(
                df=modeling_df,
                selected_metadata_path = 'feature_selection_metadata.json',
                causal_metadata_path ='causal_analysis_metadata.json'
            )

            print(f"\nPREDICTION ANALYSIS COMPLETED!")
            if results is not None:
                print(f"   Trained {len(results)} model configurations")
                print(f"   Best F1 Score: {results['Test_F1'].max():.4f}")
                print(f"   Best Sharpe Ratio: {results['Sharpe_Ratio'].max():.4f}")
        except Exception as pred_error:
            print(f"\nPrediction analysis encountered an issue: {pred_error}")
            pipeline, results = None, None

        return analyzer, causal_inf, selector,pipeline, results

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please ensure you have the required data files or adjust the data sources.")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    analyzer, causal_inf,selector, pipeline, results = main()

    if analyzer is not None:
        print("\nAnalysis completed successfully!")
        print("\nAvailable objects:")
        print("   - analyzer: UKStockAnalyzer instance with processed data")
        print("   - causal_inf: CausalInferenceAnalyzer with all methods")
        print("   - selector: FeatureSelector  with all methods")
        print("   - pipeline: Prediction pipeline (if successful)")
        print("   - results: Prediction results (if successful)")

    else:
        print("\nAnalysis failed. Please check the error messages above.")
