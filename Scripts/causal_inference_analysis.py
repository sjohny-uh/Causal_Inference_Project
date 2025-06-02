# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import networkx as nx
import dowhy
from dowhy import CausalModel
import warnings
import pyparsing
import logging
import os
if not hasattr(pyparsing, 'DelimitedList'):
    pyparsing.DelimitedList = pyparsing.delimitedList
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/Users/sheri/Downloads/final project/Causal_Inference/log/causal_inference_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CausalInferenceAnalyzer:
    """
    Causal inference analysis for UK stock market data
    """

    def __init__(self, df, target_variable='Target_Direction'):
        """
        Initialize the causal inference analyzer

        Parameters:
        ----
        df : pd.DataFrame
        The processed dataset with features and targets
        target_variable : str
        The target variable for causal analysis
        """
        self.df = df.copy()
        self.target = target_variable
        self.granger_results = pd.DataFrame()
        self.causal_features = []
        self.causal_graph = None
        self.dowhy_results = {}
        self.modeling_df = None

        # Create plots directory
        self.plots_dir = 'C:/Users/sheri/Downloads/final project/Causal_Inference/output'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            logger.info(f"Created plots directory: {self.plots_dir}")

        logger.info(f"CausalInferenceAnalyzer initialized with target variable: {target_variable}")
        logger.info(f"Dataset shape: {df.shape}")

    def check_stationarity(self, series, name):
        """
        Check if a time series is stationary using Augmented Dickey-Fuller test
        """
        try:
            result = adfuller(series.dropna())
            is_stationary = result[1] <= 0.05
            print(f"  {name}: {'Stationary' if is_stationary else 'Non-stationary'} (p-value: {result[1]:.4f})")
            logger.info(f"Stationarity test for {name}: {'Stationary' if is_stationary else 'Non-stationary'} (p-value: {result[1]:.4f})")
            return is_stationary
        except:
            print(f"  {name}: Could not test stationarity")
            logger.error(f"Could not test stationarity for {name}")
            return False

    def prepare_data_for_granger(self):
        """
        Prepare data for Granger causality tests by ensuring stationarity
        """
        print("Preparing data for Granger causality tests...")
        print("Checking stationarity of all variables:")
        logger.info("Starting data preparation for Granger causality tests")

        # Get all potential variables from the dataset
        # Exclude non-numeric and target columns
        exclude_cols = ['Date', 'Ticker', 'Target_Direction', 'Target_5D_Return',
                       'Target_10D_Return', 'Target_Excess_Return']

        # Get all numeric columns
        all_vars = [col for col in self.df.columns
                   if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]

        # Categorize variables for better organization
        macro_vars = [col for col in all_vars if any(x in col.lower() for x in
                     ['unemployment', 'cpi', 'rate', 'gbp', 'usd', 'inflation', 'gdp', 'pmi'])]

        technical_vars = [col for col in all_vars if any(x in col.lower() for x in
                         ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema', 'bollinger', 'stoch', 'williams'])]

        price_vars = [col for col in all_vars if any(x in col.lower() for x in
                     ['return', 'volatility', 'volume', 'price', 'open', 'high', 'low', 'close'])]

        # Any remaining variables
        other_vars = [col for col in all_vars
                     if col not in macro_vars + technical_vars + price_vars]

        # Combine all variables in order of priority
        test_vars = macro_vars + technical_vars + price_vars + other_vars

        print(f"Testing stationarity for {len(test_vars)} variables:")
        print(f"  - Macroeconomic: {len(macro_vars)}")
        print(f"  - Technical: {len(technical_vars)}")
        print(f"  - Price-based: {len(price_vars)}")
        print(f"  - Other: {len(other_vars)}")
        logger.info(f"Testing stationarity for {len(test_vars)} variables - Macro: {len(macro_vars)}, Technical: {len(technical_vars)}, Price: {len(price_vars)}, Other: {len(other_vars)}")

        stationary_vars = []
        non_stationary_count = 0
        error_count = 0

        for var in test_vars:
            if var in self.df.columns:
                try:
                    # Check if variable has sufficient non-null values
                    var_data = self.df[var].dropna()
                    if len(var_data) < 50:  # Minimum observations for reliable test
                        print(f"  {var}: Insufficient data ({len(var_data)} obs)")
                        logger.warning(f"Insufficient data for {var}: {len(var_data)} observations")
                        continue

                    # Test original variable
                    is_stationary = self.check_stationarity(var_data, var)
                    if is_stationary:
                        stationary_vars.append(var)
                    else:
                        non_stationary_count += 1
                        # Try first difference
                        diff_var = f"{var}_diff"
                        self.df[diff_var] = self.df[var].diff()
                        diff_data = self.df[diff_var].dropna()

                        if len(diff_data) >= 50:
                            is_diff_stationary = self.check_stationarity(diff_data, f"{var} (differenced)")
                            if is_diff_stationary:
                                stationary_vars.append(diff_var)
                            else:
                                # Try second difference for highly non-stationary series
                                diff2_var = f"{var}_diff2"
                                self.df[diff2_var] = self.df[diff_var].diff()
                                diff2_data = self.df[diff2_var].dropna()

                                if len(diff2_data) >= 50:
                                    is_diff2_stationary = self.check_stationarity(diff2_data, f"{var} (2nd diff)")
                                    if is_diff2_stationary:
                                        stationary_vars.append(diff2_var)

                except Exception as e:
                    error_count += 1
                    print(f"  {var}: Error in stationarity test - {str(e)[:50]}...")
                    logger.error(f"Error in stationarity test for {var}: {str(e)}")
                    continue

        print(f"\nStationarity test results:")
        print(f"  - Stationary variables: {len(stationary_vars)}")
        print(f"  - Non-stationary (even after differencing): {non_stationary_count - (len(stationary_vars) - len([v for v in stationary_vars if not v.endswith('_diff') and not v.endswith('_diff2')]))}")
        print(f"  - Errors encountered: {error_count}")
        logger.info(f"Stationarity test completed - Stationary: {len(stationary_vars)}, Errors: {error_count}")

        # Store stationarity results for reference
        self.stationarity_results = {
            'stationary_vars': stationary_vars,
            'total_tested': len(test_vars),
            'non_stationary_count': non_stationary_count,
            'error_count': error_count
        }

        return stationary_vars

    def granger_causality_analysis(self, max_lag=10):
        """
        Comprehensive Granger causality analysis
        """
        print(f"\\n{'='*60}")
        print("GRANGER CAUSALITY ANALYSIS")
        print(f"{'='*60}")
        logger.info("Starting Granger causality analysis")

        # Get stationary variables
        stationary_vars = self.prepare_data_for_granger()

        # Create mapping of original to stationary versions
        stationary_mapping = {}
        for var in stationary_vars:
            if var.endswith('_diff'):
                original = var.replace('_diff', '')
                stationary_mapping[original] = var
            else:
                stationary_mapping[var] = var

        # Define feature categories
        macro_features = [col for col in self.df.columns if any(x in col.lower() for x in
                         ['unemployment', 'cpi', 'rate', 'gbp', 'usd']) and 'lag' in col]

        technical_features = [col for col in self.df.columns if any(x in col.lower() for x in
                             ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema']) and 'lag' in col]

        price_features = [col for col in self.df.columns if any(x in col.lower() for x in
                         ['return', 'volatility', 'volume']) and 'lag' in col]

        all_features = macro_features + technical_features + price_features

        #Filter to only stationary features
        stationary_features = []
        for feature in all_features:
            # Check if feature itself is stationary
            if feature in stationary_vars:
                stationary_features.append(feature)
            # Check if differenced version exists and is stationary
            elif f"{feature}_diff" in stationary_vars:
                stationary_features.append(f"{feature}_diff")
            # Check base variable (remove lag suffix)
            else:
                base_var = feature.split('_lag_')[0] if '_lag_' in feature else feature
                if base_var in stationary_mapping:
                    # Reconstruct with stationary version
                    lag_part = feature.split('_lag_')[1] if '_lag_' in feature else ''
                    stationary_version = f"{stationary_mapping[base_var]}_lag_{lag_part}" if lag_part else stationary_mapping[base_var]
                    if stationary_version in self.df.columns:
                        stationary_features.append(stationary_version)

        print(f"Testing {len(stationary_features)} stationary features for Granger causality...")
        print(f"(Filtered from {len(all_features)} total features)")
        logger.info(f"Testing {len(stationary_features)} stationary features for Granger causality (filtered from {len(all_features)} total)")

        # Store results
        granger_results = []

        for feature in stationary_features:  # using only stationary features!
            if feature in self.df.columns:
                try:
                    # Create clean dataset for this pair
                    test_data = self.df[[self.target, feature]].dropna()

                    if len(test_data) > max_lag * 2:
                        # Run Granger causality test
                        gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                        # Extract p-values for each lag
                        p_values = []
                        for lag in range(1, max_lag + 1):
                            if lag in gc_result:
                                p_val = gc_result[lag][0]['ssr_chi2test'][1]
                                p_values.append(p_val)

                        # Find minimum p-value and corresponding lag
                        if p_values:
                            min_p_value = min(p_values)
                            best_lag = p_values.index(min_p_value) + 1

                            granger_results.append({
                                'feature': feature,
                                'min_p_value': min_p_value,
                                'best_lag': best_lag,
                                'significant': min_p_value < 0.05,
                                'category': 'Macro' if feature in macro_features else
                                           'Technical' if feature in technical_features else 'Price'
                            })
                            logger.debug(f"Granger test for {feature}: p-value={min_p_value:.4f}, lag={best_lag}")

                except Exception as e:
                    print(f"  Error testing {feature}: {str(e)[:50]}...")
                    logger.error(f"Error testing {feature}: {str(e)}")
                    continue

        # Convert to DataFrame and sort by significance
        self.granger_results = pd.DataFrame(granger_results)
        self.granger_results = self.granger_results.sort_values('min_p_value')

        # Identify significant causal features
        self.causal_features = self.granger_results[
            self.granger_results['significant']
        ]['feature'].tolist()

        print(f"\\n GRANGER CAUSALITY RESULTS:")
        print(f"   Total features tested: {len(granger_results)}")
        print(f"   Significant causal relationships: {len(self.causal_features)}")
        print(f"   Significance threshold: p < 0.05")
        logger.info(f"Granger causality analysis completed - {len(granger_results)} features tested, {len(self.causal_features)} significant relationships found")

        # Display top results
        print(f"\\n TOP 10 MOST SIGNIFICANT CAUSAL RELATIONSHIPS:")
        top_results = self.granger_results.head(10)
        for idx, row in top_results.iterrows():
            significance = "***" if row['min_p_value'] < 0.001 else "**" if row['min_p_value'] < 0.01 else "*"
            print(f"   {row['feature']:<30} | p-value: {row['min_p_value']:.4f} {significance} | Lag: {row['best_lag']}")

        return self.granger_results

    def build_causal_graph(self):
        """
        Build a causal graph (DAG) based on domain knowledge and Granger results
        """
        print(f"\\n{'='*60}")
        print("CAUSAL GRAPH CONSTRUCTION")
        print(f"{'='*60}")
        logger.info("Starting causal graph construction")

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes for significant causal features
        significant_features = self.granger_results[
            self.granger_results['significant']
        ]['feature'].head(15).tolist()  # Top 15 for clarity

        # Simplify feature names for visualization
        node_mapping = {}
        for feature in significant_features:
            if 'unemployment' in feature.lower():
                node_mapping[feature] = 'Unemployment'
            elif 'cpi' in feature.lower():
                node_mapping[feature] = 'Inflation'
            elif 'rate' in feature.lower():
                node_mapping[feature] = 'Interest_Rate'
            elif 'gbp' in feature.lower() or 'usd' in feature.lower():
                node_mapping[feature] = 'GBP_USD'
            elif 'rsi' in feature.lower():
                node_mapping[feature] = 'RSI'
            elif 'macd' in feature.lower():
                node_mapping[feature] = 'MACD'
            elif 'return' in feature.lower():
                node_mapping[feature] = 'Returns'
            elif 'volatility' in feature.lower():
                node_mapping[feature] = 'Volatility'
            else:
                node_mapping[feature] = feature.replace('_lag_', '_L')[:15]

        # Add target node
        node_mapping[self.target] = 'Target'

        # Add nodes
        for original, simplified in node_mapping.items():
            G.add_node(simplified, original_name=original)

        # Add edges based on Granger causality results
        for _, row in self.granger_results[self.granger_results['significant']].head(15).iterrows():
            source = node_mapping.get(row['feature'], row['feature'])
            target = node_mapping[self.target]
            G.add_edge(source, target, p_value=row['min_p_value'], lag=row['best_lag'])

        # Add domain knowledge edges (macro relationships)
        domain_edges = [
            ('Interest_Rate', 'GBP_USD'),
            ('Inflation', 'Interest_Rate'),
            ('Unemployment', 'Inflation'),
            ('GBP_USD', 'Returns'),
            ('Returns', 'Volatility'),
            ('RSI', 'MACD')  # Technical indicators correlation
        ]

        for source, target in domain_edges:
            if source in G.nodes() and target in G.nodes():
                if not G.has_edge(source, target):
                    G.add_edge(source, target, edge_type='domain_knowledge')

        self.causal_graph = G

        # Visualize the graph
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=2000, alpha=0.8)

        # Draw edges with different colors
        granger_edges = [(u, v) for u, v, d in G.edges(data=True) if 'p_value' in d]
        domain_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'domain_knowledge']

        nx.draw_networkx_edges(G, pos, edgelist=granger_edges,
                              edge_color='red', alpha=0.7, width=2, label='Granger Causal')
        nx.draw_networkx_edges(G, pos, edgelist=domain_edges,
                              edge_color='blue', alpha=0.5, style='dashed', label='Domain Knowledge')

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        plt.title('Causal Graph: UK Stock Market Drivers\\n(Red=Granger Causal, Blue=Domain Knowledge)',
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.plots_dir, 'causal_graph.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Causal graph plot saved to {plot_path}")

        # plt.show()

        print(f" Causal graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        logger.info(f"Causal graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        return G

    def dowhy_causal_analysis(self):
        """
        Use DoWhy framework for causal effect estimation and robustness checks
        """


        print(f"\\n{'='*60}")
        print("DOWHY CAUSAL ANALYSIS")
        print(f"{'='*60}")
        logger.info("Starting DoWhy causal analysis")

        # Select top 5 most significant features for detailed DoWhy analysis
        top_features = self.granger_results[
            self.granger_results['significant']
        ]['feature'].head(5).tolist()

        dowhy_results = {}

        for treatment_var in top_features:
            try:
                print(f"\\nAnalyzing causal effect of {treatment_var} on {self.target}...")
                logger.info(f"Analyzing causal effect of {treatment_var} on {self.target}")

                # Prepare data for DoWhy
                # Select relevant confounders (other significant variables)
                confounders = [f for f in top_features if f != treatment_var][:3]  # Top 3 confounders

                analysis_vars = [treatment_var, self.target] + confounders
                analysis_data = self.df[analysis_vars].dropna()

                if len(analysis_data) < 100:  # Minimum sample size
                    print(f"  Insufficient data for {treatment_var}")
                    logger.warning(f"Insufficient data for {treatment_var}: {len(analysis_data)} observations")
                    continue

                # Create causal graph string for DoWhy
                graph_str = f"digraph {{ {treatment_var} -> {self.target};"
                for confounder in confounders:
                    graph_str += f" {confounder} -> {self.target}; {confounder} -> {treatment_var};"
                graph_str += " }"

                # Build causal model
                model = CausalModel(
                    data=analysis_data,
                    treatment=treatment_var,
                    outcome=self.target,
                    graph=graph_str,
                    common_causes=confounders
                )

                # Identify causal effect
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

                # Estimate causal effect using multiple methods
                methods = [
                    "backdoor.linear_regression",
                    "backdoor.generalized_linear_model"
                ]

                estimates = {}
                for method in methods:
                    try:
                        estimate = model.estimate_effect(
                            identified_estimand,
                            method_name=method,
                            target_units="ate"
                        )
                        estimates[method] = {
                            'value': estimate.value,
                            'confidence_intervals': getattr(estimate, 'confidence_intervals', None)
                        }
                        print(f"  {method}: {estimate.value:.4f}")
                        logger.info(f"DoWhy estimate for {treatment_var} using {method}: {estimate.value:.4f}")
                    except Exception as e:
                        print(f"  {method}: Failed ({str(e)})")
                        logger.error(f"DoWhy estimation failed for {treatment_var} using {method}: {str(e)}")

                # Robustness checks
                robustness_results = {}

                # 1. Placebo test
                try:
                    placebo_data = analysis_data.copy()
                    placebo_data[f'{treatment_var}_placebo'] = np.random.permutation(
                        placebo_data[treatment_var].values
                    )

                    placebo_model = CausalModel(
                        data=placebo_data,
                        treatment=f'{treatment_var}_placebo',
                        outcome=self.target,
                        graph=graph_str.replace(treatment_var, f'{treatment_var}_placebo'),
                        common_causes=confounders
                    )

                    placebo_estimand = placebo_model.identify_effect(proceed_when_unidentifiable=True)
                    placebo_estimate = placebo_model.estimate_effect(
                        placebo_estimand,
                        method_name="backdoor.linear_regression"
                    )

                    robustness_results['placebo_test'] = placebo_estimate.value
                    print(f"  Placebo test: {placebo_estimate.value:.4f}")
                    logger.info(f"Placebo test for {treatment_var}: {placebo_estimate.value:.4f}")

                except Exception as e:
                    print(f"  Placebo test failed: {str(e)}")
                    logger.error(f"Placebo test failed for {treatment_var}: {str(e)}")

                # 2. Random common cause test
                try:
                    refutation = model.refute_estimate(
                        identified_estimand,
                        estimates['backdoor.linear_regression'] if 'backdoor.linear_regression' in estimates else list(estimates.values())[0],
                        method_name="random_common_cause"
                    )
                    robustness_results['random_common_cause'] = refutation.new_effect
                    print(f"  Random common cause test: {refutation.new_effect:.4f}")
                    logger.info(f"Random common cause test for {treatment_var}: {refutation.new_effect:.4f}")

                except Exception as e:
                    print(f"  Random common cause test failed: {str(e)}")
                    logger.error(f"Random common cause test failed for {treatment_var}: {str(e)}")

                dowhy_results[treatment_var] = {
                    'estimates': estimates,
                    'robustness': robustness_results,
                    'sample_size': len(analysis_data)
                }

            except Exception as e:
                print(f"  DoWhy analysis failed for {treatment_var}: {str(e)}")
                logger.error(f"DoWhy analysis failed for {treatment_var}: {str(e)}")
                continue

        self.dowhy_results = dowhy_results
        logger.info(f"DoWhy causal analysis completed for {len(dowhy_results)} variables")
        return dowhy_results

    def sensitivity_analysis(self):
        """
        Perform sensitivity analysis for causal relationships
        """
        print(f"\\n{'='*60}")
        print("SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        logger.info("Starting sensitivity analysis")

        # Test stability across different time periods
        print("Testing temporal stability of causal relationships...")
        logger.info("Testing temporal stability of causal relationships")

        # Split data into periods
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        periods = [
            ('Pre-COVID', self.df['Date'] < '2020-03-01'),
            ('COVID Period', (self.df['Date'] >= '2020-03-01') & (self.df['Date'] < '2021-06-01')),
            ('Post-COVID', self.df['Date'] >= '2021-06-01')
        ]

        stability_results = {}

        for period_name, period_mask in periods:
            period_data = self.df[period_mask]
            if len(period_data) < 100:  # Minimum sample size
                logger.warning(f"Insufficient data for {period_name}: {len(period_data)} observations")
                continue

            print(f"\\n  {period_name}: {len(period_data)} observations")
            logger.info(f"Analyzing {period_name}: {len(period_data)} observations")

            # Test top 5 causal relationships in this period
            period_results = []
            for feature in self.causal_features[:5]:
                if feature in period_data.columns:
                    try:
                        test_data = period_data[[self.target, feature]].dropna()
                        if len(test_data) > 20:
                            gc_result = grangercausalitytests(test_data, maxlag=5, verbose=False)
                            min_p = min([gc_result[lag][0]['ssr_chi2test'][1] for lag in range(1, 6)])
                            period_results.append((feature, min_p))
                            logger.debug(f"{period_name} - {feature}: p-value={min_p:.4f}")
                    except:
                        continue

            stability_results[period_name] = period_results

            # Show top 3 results for this period
            period_results.sort(key=lambda x: x[1])
            for feature, p_val in period_results[:3]:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {feature:<25} p-value: {p_val:.4f} {significance}")

        logger.info("Sensitivity analysis completed")
        return stability_results

    def prepare_modeling_dataset(self):
        """
        Prepare final dataset for modeling with causal insights
        """
        print(f"\\n{'='*60}")
        print("PREPARING MODELING DATASET")
        print(f"{'='*60}")
        logger.info("Starting modeling dataset preparation")

        # Start with original dataframe
        self.modeling_df = self.df.copy()

        # Add causal feature indicators and importance scores
        if hasattr(self, 'causal_features') and self.causal_features:
            print(f"Found {len(self.causal_features)} causal features")
            logger.info(f"Found {len(self.causal_features)} causal features")

            # Create feature importance scores based on Granger results
            if hasattr(self, 'granger_results') and not self.granger_results.empty:
                # Add causal importance scores
                causal_scores = {}
                for _, row in self.granger_results.iterrows():
                    # Convert p-value to importance score (lower p-value = higher importance)
                    importance = -np.log10(max(row['min_p_value'], 1e-10))
                    causal_scores[row['feature']] = importance

                # Add causal importance as metadata (not as features to avoid data leakage)
                self.modeling_df.attrs['causal_importance_scores'] = causal_scores

            # Categorize features
            all_features = [col for col in self.modeling_df.columns
                           if col not in ['Date', 'Ticker', 'Target_Direction', 'Target_5D_Return',
                                         'Target_10D_Return', 'Target_Excess_Return']]

            causal_feature_names = self.causal_features
            non_causal_features = [f for f in all_features if f not in causal_feature_names]

            # Categorize causal features by type
            macro_causal = [f for f in causal_feature_names if any(x in f.lower() for x in
                           ['unemployment', 'cpi', 'rate', 'gbp', 'usd'])]
            technical_causal = [f for f in causal_feature_names if any(x in f.lower() for x in
                               ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema'])]
            price_causal = [f for f in causal_feature_names if any(x in f.lower() for x in
                           ['return', 'volatility', 'volume'])]

            print(f"Feature breakdown:")
            print(f"   Total features: {len(all_features)}")
            print(f"   Causal features: {len(causal_feature_names)}")
            print(f"    - Macroeconomic: {len(macro_causal)}")
            print(f"    - Technical: {len(technical_causal)}")
            print(f"    - Price-based: {len(price_causal)}")
            print(f"   Non-causal features: {len(non_causal_features)}")
            logger.info(f"Feature breakdown - Total: {len(all_features)}, Causal: {len(causal_feature_names)}, Non-causal: {len(non_causal_features)}")

            # Store feature lists as metadata
            self.modeling_df.attrs['causal_features'] = causal_feature_names
            self.modeling_df.attrs['non_causal_features'] = non_causal_features
            self.modeling_df.attrs['macro_causal_features'] = macro_causal
            self.modeling_df.attrs['technical_causal_features'] = technical_causal
            self.modeling_df.attrs['price_causal_features'] = price_causal

            # Add feature priority rankings
            if hasattr(self, 'granger_results') and not self.granger_results.empty:
                # Create priority rankings based on p-values
                priority_features = self.granger_results.sort_values('min_p_value')['feature'].tolist()
                self.modeling_df.attrs['feature_priority_ranking'] = priority_features

        else:
            print("Warning: No causal features identified")
            logger.warning("No causal features identified")
            self.modeling_df.attrs['causal_features'] = []
            self.modeling_df.attrs['non_causal_features'] = []

        # Add analysis metadata
        self.modeling_df.attrs['analysis_date'] = pd.Timestamp.now()
        self.modeling_df.attrs['target_variable'] = self.target
        self.modeling_df.attrs['granger_significance_threshold'] = 0.05

        if hasattr(self, 'granger_results') and not self.granger_results.empty:
            self.modeling_df.attrs['total_features_tested'] = len(self.granger_results)
            self.modeling_df.attrs['significant_relationships'] = len(self.causal_features)

        # Save the enhanced dataset
        output_filename = 'causal_enhanced_modeling_dataset.csv'
        self.modeling_df.to_csv(output_filename, index=False)
        print(f"Enhanced dataset saved: {output_filename}")
        logger.info(f"Enhanced dataset saved: {output_filename}")

        # Save feature metadata separately
        metadata = {
            'causal_features': self.modeling_df.attrs.get('causal_features', []),
            'non_causal_features': self.modeling_df.attrs.get('non_causal_features', []),
            'macro_causal_features': self.modeling_df.attrs.get('macro_causal_features', []),
            'technical_causal_features': self.modeling_df.attrs.get('technical_causal_features', []),
            'price_causal_features': self.modeling_df.attrs.get('price_causal_features', []),
            'feature_priority_ranking': self.modeling_df.attrs.get('feature_priority_ranking', []),
            'causal_importance_scores': self.modeling_df.attrs.get('causal_importance_scores', {}),
            'target_variable': self.target,
            'analysis_summary': {
                'total_features_tested': self.modeling_df.attrs.get('total_features_tested', 0),
                'significant_relationships': self.modeling_df.attrs.get('significant_relationships', 0),
                'dataset_shape': self.modeling_df.shape,
                'analysis_date': str(self.modeling_df.attrs.get('analysis_date', ''))
            }
        }

        metadata_filename = 'causal_analysis_metadata.json'
        import json
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Feature metadata saved: {metadata_filename}")
        logger.info(f"Feature metadata saved: {metadata_filename}")

        return self.modeling_df

    def generate_causal_report(self):
        """
        Generate comprehensive causal inference report
        """
        print(f"\\n{'='*80}")
        print("CAUSAL INFERENCE ANALYSIS - COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        logger.info("Generating comprehensive causal inference report")

        # Summary statistics
        print(f"\\n ANALYSIS SUMMARY:")
        print(f"   Dataset size: {len(self.df):,} observations")
        print(f"   Target variable: {self.target}")
        print(f"   Features tested: {len(self.granger_results) if hasattr(self, 'granger_results') else 'N/A'}")
        print(f"   Significant causal relationships: {len(self.causal_features)}")
        logger.info(f"Analysis summary - Dataset: {len(self.df):,} obs, Target: {self.target}, Significant relationships: {len(self.causal_features)}")

        # Top causal drivers
        if hasattr(self, 'granger_results') and not self.granger_results.empty:
            print(f"\\n TOP CAUSAL DRIVERS (Granger Causality):")
            top_drivers = self.granger_results[self.granger_results['significant']].head(10)

            for idx, (_, row) in enumerate(top_drivers.iterrows(), 1):
                print(f"   {idx:2d}. {row['feature']:<30} | p-value: {row['min_p_value']:.4f} | Lag: {row['best_lag']} days")

        # DoWhy results summary
        if hasattr(self, 'dowhy_results') and self.dowhy_results:
            print(f"\\n CAUSAL EFFECT ESTIMATES (DoWhy):")
            for treatment, results in self.dowhy_results.items():
                print(f"\\n   Treatment: {treatment}")
                for method, estimate in results['estimates'].items():
                    print(f"    {method}: {estimate['value']:.4f}")

                if 'robustness' in results:
                    print(f"   Robustness checks:")
                    for test, value in results['robustness'].items():
                        print(f"    {test}: {value:.4f}")

        # Recommendations for modeling
        print(f"\\n RECOMMENDATIONS FOR MODELING:")

        if self.causal_features:
            macro_features = [f for f in self.causal_features if any(x in f.lower() for x in ['unemployment', 'cpi', 'rate', 'gbp'])]
            technical_features = [f for f in self.causal_features if any(x in f.lower() for x in ['rsi', 'macd', 'bb_', 'atr'])]

            print(f"   1. Macroeconomic drivers ({len(macro_features)} features):")
            for feature in macro_features:
                print(f"    {feature}")

            print(f"   2. Technical drivers ({len(technical_features)} features):")
            for feature in technical_features:
                print(f"    {feature}")

        logger.info("Causal inference report generation completed")

        return {
            'causal_features': self.causal_features,
            'granger_results': self.granger_results if hasattr(self, 'granger_results') else None,
            'dowhy_results': self.dowhy_results if hasattr(self, 'dowhy_results') else None,
            'modeling_df': self.modeling_df
        }

# Usage example function
def run_causal_analysis(df, target='Target_Direction'):
    """
    Run complete causal inference analysis pipeline

    Parameters:
    ----
    df : pd.DataFrame
    Processed dataset with features and targets
    target : str
    Target variable name

    Returns:
    ----
    analyzer : CausalInferenceAnalyzer
    Analyzer object with results
    modeling_df : pd.DataFrame
    Final dataset ready for modeling with causal insights
    """
    print(" STARTING  CAUSAL INFERENCE ANALYSIS")
    print("="*80)
    logger.info("Starting  causal inference analysis")

    # Initialize analyzer
    analyzer = CausalInferenceAnalyzer(df, target)

    # Step 1: Granger Causality Analysis
    granger_results = analyzer.granger_causality_analysis(max_lag=10)

    # Step 2: Build Causal Graph
    causal_graph = analyzer.build_causal_graph()

    # Step 3: DoWhy Analysis 
    dowhy_results = analyzer.dowhy_causal_analysis()

    # Step 4: Sensitivity Analysis
    sensitivity_results = analyzer.sensitivity_analysis()

    # Step 5: Prepare Modeling Dataset
    modeling_df = analyzer.prepare_modeling_dataset()

    # Step 6: Generate Report
    final_report = analyzer.generate_causal_report()

    print(f"\\n CAUSAL INFERENCE ANALYSIS COMPLETED!")
    print(f"   Identified {len(analyzer.causal_features)} causally significant features")
    print(f"   Modeling dataset ready with {modeling_df.shape[0]} observations and {modeling_df.shape[1]} features")
    logger.info(f"Causal inference analysis completed - {len(analyzer.causal_features)} significant features identified")

    return analyzer, modeling_df

def run_causal_analysis_with_modeling_output(df, target='Target_Direction'):
    """
    Function that returns both analyzer and modeling-ready dataframe

    Parameters:
    ----
    df : pd.DataFrame
    Processed dataset with features and targets
    target : str
    Target variable name

    Returns:
    ----
    analyzer : CausalInferenceAnalyzer
    Analyzer object with all results
    modeling_df : pd.DataFrame
    Final dataset ready for modeling
    """
    logger.info("Running causal analysis with modeling output")
    analyzer, modeling_df = run_causal_analysis(df, target)
    return analyzer, modeling_df
