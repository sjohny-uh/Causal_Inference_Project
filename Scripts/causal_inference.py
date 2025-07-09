import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.multitest import multipletests
import warnings
import json
import logging
import os

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CausalInferenceAnalyzer:
    def __init__(self, df, target_variable='Target_Direction'):
        self.df = df.copy()
        self.target = target_variable
        self.granger_results = pd.DataFrame()
        self.causal_features = []
        self.modeling_df = None

        self.plots_dir = 'C:/Users/sheri/Downloads/Causal_Inference_Project/Output'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        logger.info(f"CausalInferenceAnalyzer initialized with target: {target_variable}")
        logger.info(f"Dataset shape: {df.shape}")

    def check_stationarity_batch(self, variables):
        """Batch check stationarity for multiple variables"""
        stationary_vars = []

        for var in variables:
            if var not in self.df.columns:
                continue

            try:
                series = self.df[var].dropna()
                if len(series) < 50:
                    continue

                # Test original series
                adf_stat, p_value = adfuller(series)[:2]
                if p_value <= 0.05:
                    stationary_vars.append(var)
                else:
                    # Try first difference
                    diff_series = series.diff().dropna()
                    if len(diff_series) >= 50:
                        adf_stat_diff, p_value_diff = adfuller(diff_series)[:2]
                        if p_value_diff <= 0.05:
                            diff_var = f"{var}_diff"
                            self.df[diff_var] = self.df[var].diff()
                            stationary_vars.append(diff_var)
            except:
                continue

        return stationary_vars

    def granger_causality_analysis(self, max_lag=5):
        """Simplified Granger causality analysis"""
        print(f"\nGRANGER CAUSALITY ANALYSIS")
        print(f"{'='*50}")

        # Get potential features
        exclude_cols = ['Date', 'Ticker', 'Target_Direction', 'Target_5D_Return',
                       'Target_10D_Return', 'Target_Excess_Return']

        all_vars = [col for col in self.df.columns
                   if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]

        # Check stationarity
        print(f"Checking stationarity for {len(all_vars)} variables...")
        stationary_vars = self.check_stationarity_batch(all_vars)
        print(f"Found {len(stationary_vars)} stationary variables")

        # Run Granger tests
        results = []
        for feature in stationary_vars:
            try:
                test_data = self.df[[self.target, feature]].dropna()
                if len(test_data) <= max_lag * 2:
                    continue

                gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                # Get minimum p-value across lags
                p_values = []
                for lag in range(1, max_lag + 1):
                    if lag in gc_result:
                        p_val = gc_result[lag][0]['ssr_chi2test'][1]
                        p_values.append(p_val)

                if p_values:
                    min_p_value = min(p_values)
                    best_lag = p_values.index(min_p_value) + 1

                    # Categorize feature
                    if any(x in feature.lower() for x in ['unemployment', 'cpi', 'rate', 'gbp', 'usd']):
                        category = 'Macro'
                    elif any(x in feature.lower() for x in ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema']):
                        category = 'Technical'
                    elif any(x in feature.lower() for x in ['return', 'volatility', 'volume']):
                        category = 'Price'
                    else:
                        category = 'Other'

                    results.append({
                        'feature': feature,
                        'min_p_value': min_p_value,
                        'best_lag': best_lag,
                        'category': category,
                        'sample_size': len(test_data)
                    })
            except Exception as e:
                logger.error(f"Error testing {feature}: {str(e)}")
                continue

        # Convert to DataFrame
        self.granger_results = pd.DataFrame(results)

        if not self.granger_results.empty:
            # Apply multiple testing correction
            _, corrected_p_values, _, _ = multipletests(
                self.granger_results['min_p_value'],
                alpha=0.05,
                method='fdr_bh'
            )

            self.granger_results['corrected_p_value'] = corrected_p_values
            self.granger_results['significant'] = corrected_p_values < 0.05

            # Sort by corrected p-value
            self.granger_results = self.granger_results.sort_values('corrected_p_value')

            # Get significant causal features
            self.causal_features = self.granger_results[
                self.granger_results['significant']
            ]['feature'].tolist()

        print(f"\nResults:")
        print(f"   Total features tested: {len(results)}")
        print(f"   Significant causal relationships: {len(self.causal_features)}")

        # Display top results
        if not self.granger_results.empty:
            print(f"\nTop 10 Most Significant:")
            top_results = self.granger_results.head(10)
            for idx, row in top_results.iterrows():
                sig_marker = "***" if row['corrected_p_value'] < 0.001 else "**" if row['corrected_p_value'] < 0.01 else "*"
                print(f"   {row['feature']:<30} | p-value: {row['corrected_p_value']:.4f} {sig_marker}")

        return self.granger_results

    def prepare_modeling_dataset(self):
        """Prepare final dataset for modeling"""
        print(f"\nPREPARING MODELING DATASET")
        print(f"{'='*40}")

        self.modeling_df = self.df.copy()

        # Store metadata
        self.modeling_df.attrs['causal_features'] = self.causal_features
        self.modeling_df.attrs['target_variable'] = self.target

        # Create importance scores from Granger results
        if not self.granger_results.empty:
            importance_scores = {}
            for _, row in self.granger_results.iterrows():
                # Convert p-value to importance score
                importance_scores[row['feature']] = -np.log10(max(row['corrected_p_value'], 1e-10))

            self.modeling_df.attrs['causal_importance_scores'] = importance_scores

        print(f"Modeling dataset prepared:")
        print(f"   Shape: {self.modeling_df.shape}")
        print(f"   Causal features: {len(self.causal_features)}")

        return self.modeling_df

    def save_causal_metadata(self, filepath='causal_analysis_metadata.json'):
        """Save causal analysis metadata in original format"""
        print(f"\nSaving metadata to {filepath}...")

        # Categorize causal features
        macro_causal_features = []
        technical_causal_features = []
        price_causal_features = []
        other_causal_features = []

        for feature in self.causal_features:
            if any(x in feature.lower() for x in ['unemployment', 'cpi', 'rate', 'gbp', 'usd']):
                macro_causal_features.append(feature)
            elif any(x in feature.lower() for x in ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema']):
                technical_causal_features.append(feature)
            elif any(x in feature.lower() for x in ['return', 'volatility', 'volume']):
                price_causal_features.append(feature)
            else:
                other_causal_features.append(feature)

        # Get all features
        all_features = [col for col in self.df.columns
                       if col not in ['Date', 'Ticker', 'Target_Direction', 'Target_5D_Return',
                                     'Target_10D_Return', 'Target_Excess_Return']]
        non_causal_features = [f for f in all_features if f not in self.causal_features]

        # Feature priority ranking
        if not self.granger_results.empty:
            feature_priority_ranking = self.granger_results.sort_values('corrected_p_value')['feature'].tolist()
            causal_importance_scores = {}
            for _, row in self.granger_results.iterrows():
                causal_importance_scores[row['feature']] = -np.log10(max(row['corrected_p_value'], 1e-10))
        else:
            feature_priority_ranking = []
            causal_importance_scores = {}

        # Store in modeling_df.attrs
        self.modeling_df.attrs['causal_features'] = self.causal_features
        self.modeling_df.attrs['non_causal_features'] = non_causal_features
        self.modeling_df.attrs['macro_causal_features'] = macro_causal_features
        self.modeling_df.attrs['technical_causal_features'] = technical_causal_features
        self.modeling_df.attrs['price_causal_features'] = price_causal_features
        self.modeling_df.attrs['other_causal_features'] = other_causal_features
        self.modeling_df.attrs['feature_priority_ranking'] = feature_priority_ranking
        self.modeling_df.attrs['causal_importance_scores'] = causal_importance_scores
        self.modeling_df.attrs['total_features_tested'] = len(self.granger_results) if hasattr(self, 'granger_results') else 0
        self.modeling_df.attrs['significant_relationships'] = len(self.causal_features)
        self.modeling_df.attrs['analysis_date'] = pd.Timestamp.now().isoformat()

        # Create metadata in original format
        metadata = {
            'causal_features': self.causal_features,
            'non_causal_features': non_causal_features,
            'macro_causal_features': macro_causal_features,
            'technical_causal_features': technical_causal_features,
            'price_causal_features': price_causal_features,
            'other_causal_features': other_causal_features,
            'feature_priority_ranking': feature_priority_ranking,
            'causal_importance_scores': causal_importance_scores,
            'target_variable': self.target,
            'modeling_approach': 'simplified_granger',
            'analysis_summary': {
                'total_features_tested': len(self.granger_results) if hasattr(self, 'granger_results') else 0,
                'significant_relationships': len(self.causal_features),
                'dataset_shape': list(self.modeling_df.shape),
                'analysis_date': str(pd.Timestamp.now().isoformat())
            }
        }

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Metadata saved successfully!")
        print(f"   Causal features: {len(self.causal_features)}")
        print(f"   Non-causal features: {len(non_causal_features)}")

        return metadata

    def run_complete_analysis(self):
        """Run complete simplified causal analysis"""
        print(f"\n{'='*60}")
        print("RUNNING SIMPLIFIED CAUSAL INFERENCE ANALYSIS")
        print(f"{'='*60}")

        try:
            # Step 1: Granger Causality Analysis
            self.granger_causality_analysis()

            # Step 2: Prepare Modeling Dataset
            self.prepare_modeling_dataset()

            # Step 3: Save Metadata
            self.save_causal_metadata()

            print(f"\n{'='*60}")
            print("CAUSAL INFERENCE ANALYSIS COMPLETED!")
            print(f"{'='*60}")

        except Exception as e:
            print(f"\nERROR: {str(e)}")
            logger.error(f"Analysis failed: {str(e)}")
            raise

        return {
            'granger_results': self.granger_results,
            'causal_features': self.causal_features,
            'modeling_df': self.modeling_df
        }

def run_causal_analysis(df, target_variable='Target_Direction'):
    """Run simplified causal inference analysis"""
    print("Initializing Causal Inference Analyzer...")
    analyzer = CausalInferenceAnalyzer(df, target_variable)

    print("Running analysis pipeline...")
    results = analyzer.run_complete_analysis()

    return analyzer, results

def extract_top_causal_features(analyzer, n_features=20):
    """Extract top N causal features"""
    if hasattr(analyzer, 'causal_features'):
        return analyzer.causal_features[:n_features]
    return []

def get_feature_importance_scores(analyzer):
    """Get feature importance scores"""
    if hasattr(analyzer, 'modeling_df') and analyzer.modeling_df is not None:
        return analyzer.modeling_df.attrs.get('causal_importance_scores', {})
    return {}

if __name__ == "__main__":
    print("Simplified Causal Inference Analysis Script Ready!")
    print("Usage: analyzer, results = run_causal_analysis(df)")
