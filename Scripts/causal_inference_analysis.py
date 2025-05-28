# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    print("DoWhy not available. Install with: pip install dowhy")
    DOWHY_AVAILABLE = False

class CausalInferenceAnalyzer:
    """
    Causal inference analysis for UK stock market data
    """

    def __init__(self, df, target_variable='Target_Direction'):
        """
        Initialize the causal inference analyzer

        Parameters:
        -----------
        df : pd.DataFrame
            The processed dataset with features and targets
        target_variable : str
            The target variable for causal analysis
        """
        self.df = df.copy()
        self.target = target_variable
        self.granger_results = {}
        self.causal_features = []
        self.causal_graph = None
        self.dowhy_results = {}

    def check_stationarity(self, series, name):
        """
        Check if a time series is stationary using Augmented Dickey-Fuller test
        """
        try:
            result = adfuller(series.dropna())
            is_stationary = result[1] <= 0.05
            print(f"  {name}: {'Stationary' if is_stationary else 'Non-stationary'} (p-value: {result[1]:.4f})")
            return is_stationary
        except:
            print(f"  {name}: Could not test stationarity")
            return False

    def prepare_data_for_granger(self):
        """
        Prepare data for Granger causality tests by ensuring stationarity
        """
        print("Preparing data for Granger causality tests...")
        print("Checking stationarity of key variables:")

        # Test stationarity of key variables
        key_vars = ['Daily_Return', 'UnemploymentRate', 'CPI', 'Rate', 'GBP/USD', 'RSI_14', 'MACD']
        stationary_vars = []

        for var in key_vars:
            if var in self.df.columns:
                is_stationary = self.check_stationarity(self.df[var], var)
                if is_stationary:
                    stationary_vars.append(var)
                else:
                    # Try first difference
                    diff_var = f"{var}_diff"
                    self.df[diff_var] = self.df[var].diff()
                    is_diff_stationary = self.check_stationarity(self.df[diff_var], f"{var} (differenced)")
                    if is_diff_stationary:
                        stationary_vars.append(diff_var)

        print(f" {len(stationary_vars)} stationary variables identified")
        return stationary_vars

    def granger_causality_analysis(self, max_lag=10):
        """
        Comprehensive Granger causality analysis
        """
        print(f"\n{'='*60}")
        print("GRANGER CAUSALITY ANALYSIS")
        print(f"{'='*60}")

        # Prepare stationary data
        stationary_vars = self.prepare_data_for_granger()

        # Define feature categories for systematic testing
        macro_features = [col for col in self.df.columns if any(x in col.lower() for x in
                         ['unemployment', 'cpi', 'rate', 'gbp', 'usd']) and 'lag' in col]

        technical_features = [col for col in self.df.columns if any(x in col.lower() for x in
                             ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema']) and 'lag' in col]

        price_features = [col for col in self.df.columns if any(x in col.lower() for x in
                         ['return', 'volatility', 'volume']) and 'lag' in col]

        all_features = macro_features + technical_features + price_features

        print(f"Testing {len(all_features)} lagged features for Granger causality...")

        # Store results
        granger_results = []

        for feature in all_features:
            if feature in self.df.columns:
                try:
                    # Create clean dataset for this pair
                    test_data = self.df[[self.target, feature]].dropna()

                    if len(test_data) > max_lag * 2:  # Ensure sufficient data
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

                except Exception as e:
                    print(f"  Error testing {feature}: {str(e)[:50]}...")
                    continue

        # Convert to DataFrame and sort by significance
        self.granger_results = pd.DataFrame(granger_results)
        self.granger_results = self.granger_results.sort_values('min_p_value')

        # Identify significant causal features
        self.causal_features = self.granger_results[
            self.granger_results['significant']
        ]['feature'].tolist()

        print(f"\n GRANGER CAUSALITY RESULTS:")
        print(f"   • Total features tested: {len(granger_results)}")
        print(f"   • Significant causal relationships: {len(self.causal_features)}")
        print(f"   • Significance threshold: p < 0.05")

        # Display top results
        print(f"\n TOP 10 MOST SIGNIFICANT CAUSAL RELATIONSHIPS:")
        top_results = self.granger_results.head(10)
        for idx, row in top_results.iterrows():
            significance = "***" if row['min_p_value'] < 0.001 else "**" if row['min_p_value'] < 0.01 else "*"
            print(f"   {row['feature']:<30} | p-value: {row['min_p_value']:.4f} {significance} | Lag: {row['best_lag']}")

        return self.granger_results

    def build_causal_graph(self):
        """
        Build a causal graph (DAG) based on domain knowledge and Granger results
        """
        print(f"\n{'='*60}")
        print("CAUSAL GRAPH CONSTRUCTION")
        print(f"{'='*60}")

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

        plt.title('Causal Graph: UK Stock Market Drivers\n(Red=Granger Causal, Blue=Domain Knowledge)',
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print(f" Causal graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        return G

    def dowhy_causal_analysis(self):
        """
        Use DoWhy framework for causal effect estimation and robustness checks
        """
        if not DOWHY_AVAILABLE:
            print("DoWhy not available. Skipping DoWhy analysis.")
            return {}

        print(f"\n{'='*60}")
        print("DOWHY CAUSAL ANALYSIS")
        print(f"{'='*60}")

        # Select top 5 most significant features for detailed DoWhy analysis
        top_features = self.granger_results[
            self.granger_results['significant']
        ]['feature'].head(5).tolist()

        dowhy_results = {}

        for treatment_var in top_features:
            try:
                print(f"\nAnalyzing causal effect of {treatment_var} on {self.target}...")

                # Prepare data for DoWhy
                # Select relevant confounders (other significant variables)
                confounders = [f for f in top_features if f != treatment_var][:3]  # Top 3 confounders

                analysis_vars = [treatment_var, self.target] + confounders
                analysis_data = self.df[analysis_vars].dropna()

                if len(analysis_data) < 100:  # Minimum sample size
                    print(f"  Insufficient data for {treatment_var}")
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
                    "backdoor.propensity_score_matching",
                    "backdoor.linear_regression"
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
                    except Exception as e:
                        print(f"  {method}: Failed ({str(e)[:30]}...)")

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

                except Exception as e:
                    print(f"  Placebo test failed: {str(e)[:30]}...")

                # 2. Random common cause test
                try:
                    refutation = model.refute_estimate(
                        identified_estimand,
                        estimates['backdoor.linear_regression'] if 'backdoor.linear_regression' in estimates else list(estimates.values())[0],
                        method_name="random_common_cause"
                    )
                    robustness_results['random_common_cause'] = refutation.new_effect
                    print(f"  Random common cause test: {refutation.new_effect:.4f}")

                except Exception as e:
                    print(f"  Random common cause test failed: {str(e)[:30]}...")

                dowhy_results[treatment_var] = {
                    'estimates': estimates,
                    'robustness': robustness_results,
                    'sample_size': len(analysis_data)
                }

            except Exception as e:
                print(f"  DoWhy analysis failed for {treatment_var}: {str(e)[:50]}...")
                continue

        self.dowhy_results = dowhy_results
        return dowhy_results

    def sensitivity_analysis(self):
        """
        Perform sensitivity analysis for causal relationships
        """
        print(f"\n{'='*60}")
        print("SENSITIVITY ANALYSIS")
        print(f"{'='*60}")

        # Test stability across different time periods
        print("Testing temporal stability of causal relationships...")

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
                continue

            print(f"\n  {period_name}: {len(period_data)} observations")

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
                    except:
                        continue

            stability_results[period_name] = period_results

            # Show top 3 results for this period
            period_results.sort(key=lambda x: x[1])
            for feature, p_val in period_results[:3]:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {feature:<25} p-value: {p_val:.4f} {significance}")

        return stability_results

    def generate_causal_report(self):
        """
        Generate comprehensive causal inference report
        """
        print(f"\n{'='*80}")
        print("CAUSAL INFERENCE ANALYSIS - COMPREHENSIVE REPORT")
        print(f"{'='*80}")

        # Summary statistics
        print(f"\n ANALYSIS SUMMARY:")
        print(f"   • Dataset size: {len(self.df):,} observations")
        print(f"   • Target variable: {self.target}")
        print(f"   • Features tested: {len(self.granger_results) if hasattr(self, 'granger_results') else 'N/A'}")
        print(f"   • Significant causal relationships: {len(self.causal_features)}")

        # Top causal drivers
        if hasattr(self, 'granger_results') and not self.granger_results.empty:
            print(f"\n TOP CAUSAL DRIVERS (Granger Causality):")
            top_drivers = self.granger_results[self.granger_results['significant']].head(10)

            for idx, (_, row) in enumerate(top_drivers.iterrows(), 1):
                print(f"   {idx:2d}. {row['feature']:<30} | p-value: {row['min_p_value']:.4f} | Lag: {row['best_lag']} days")

        # DoWhy results summary
        if hasattr(self, 'dowhy_results') and self.dowhy_results:
            print(f"\n CAUSAL EFFECT ESTIMATES (DoWhy):")
            for treatment, results in self.dowhy_results.items():
                print(f"\n   Treatment: {treatment}")
                for method, estimate in results['estimates'].items():
                    print(f"     {method}: {estimate['value']:.4f}")

                if 'robustness' in results:
                    print(f" Robustness checks:")
                    for test, value in results['robustness'].items():
                        print(f"       {test}: {value:.4f}")

        # Recommendations
        print(f"\n RECOMMENDATIONS FOR MODELING:")

        if self.causal_features:
            macro_features = [f for f in self.causal_features if any(x in f.lower() for x in ['unemployment', 'cpi', 'rate', 'gbp'])]
            technical_features = [f for f in self.causal_features if any(x in f.lower() for x in ['rsi', 'macd', 'bb_', 'atr'])]

            print(f"   1. Macroeconomic drivers ({len(macro_features)} features):")
            for feature in macro_features[:5]:
                print(f"      • {feature}")

            print(f"   2. Technical drivers ({len(technical_features)} features):")
            for feature in technical_features[:5]:
                print(f"      • {feature}")

            

        return {
            'causal_features': self.causal_features,
            'granger_results': self.granger_results if hasattr(self, 'granger_results') else None,
            'dowhy_results': self.dowhy_results if hasattr(self, 'dowhy_results') else None
        }
''''
# Usage example function
def run_causal_analysis(df, target='Target_Direction'):
    """
    Run complete causal inference analysis pipeline

    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataset with features and targets
    target : str
        Target variable name

    Returns:
    --------
    CausalInferenceAnalyzer: Analyzer object with results
    """
    print(" STARTING COMPREHENSIVE CAUSAL INFERENCE ANALYSIS")
    print("="*80)

    # Initialize analyzer
    analyzer = CausalInferenceAnalyzer(df, target)

    # Step 1: Granger Causality Analysis
    granger_results = analyzer.granger_causality_analysis(max_lag=10)

    # Step 2: Build Causal Graph
    causal_graph = analyzer.build_causal_graph()

    # Step 3: DoWhy Analysis (if available)
    if DOWHY_AVAILABLE:
        dowhy_results = analyzer.dowhy_causal_analysis()

    # Step 4: Sensitivity Analysis
    sensitivity_results = analyzer.sensitivity_analysis()

    # Step 5: Generate Report
    final_report = analyzer.generate_causal_report()

    print(f"\n CAUSAL INFERENCE ANALYSIS COMPLETED!")
    print(f"   Identified {len(analyzer.causal_features)} causally significant features")

    return analyzer

'''
