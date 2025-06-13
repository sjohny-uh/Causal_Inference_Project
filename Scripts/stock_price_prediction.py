import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class RealDataPredictiveModelingPipeline:
    """
    Predictive modeling pipeline  from causal inference step
    """

    def __init__(self, df):
        if df is None:
            raise ValueError("DataFrame is required.")
        self.df = df
        self.results_df = None

    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        print("LOADING AND PREPARING data")
        print("-" * 50)

        self.data = self.df.copy()
        print(f"Using provided dataframe: {len(self.data)} records")

        # Prepare target variable
        self.target = 'Target_Direction'

        # Data quality check
        print(f"\\n Data Quality Check:")
        print(f"   • Dataset shape: {self.data.shape}")
        print(f"   • Available columns: {list(self.data.columns)}")

        if self.target in self.data.columns:
            print(f"   • Target distribution: {self.data[self.target].value_counts().to_dict()}")
            print(f"   • Missing values in target: {self.data[self.target].isna().sum()}")

            # Remove rows with missing target
            initial_len = len(self.data)
            self.data = self.data.dropna(subset=[self.target])
            final_len = len(self.data)

            if initial_len != final_len:
                print(f"   • Removed {initial_len - final_len} rows with missing target")
        else:
            raise ValueError(f"Target column '{self.target}' not found in the data. Available columns: {list(self.data.columns)}")

        # Check if we have enough data
        if len(self.data) < 100:
            print(f"  Warning: Only {len(self.data)} samples available. Results may not be reliable.")

        return self.data

    def prepare_feature_sets(self):
        """Prepare different feature sets for comparison using data"""
        print(f"\\n PREPARING FEATURE SETS FROM data")
        print("-" * 50)

        # Exclude non-feature columns
        exclude_cols = ['Date', 'Ticker', 'Target_Direction', 'Target_Excess_Return',
                       'Target_5D_Return', 'Target_10D_Return']

        # Get all available features
        available_features = [col for col in self.data.columns if col not in exclude_cols]

        if len(available_features) == 0:
            raise ValueError("No features found in the data after excluding target and metadata columns.")

        print(f"Available features: {len(available_features)}")
        print(f"   Features: {available_features[:10]}{'...' if len(available_features) > 10 else ''}")

        # Create feature sets
        feature_sets = {}

        # 1. All features
        feature_sets['all_features'] = {
            'features': available_features,
            'description': 'All available features'
        }

        # 2. Lag features only (causal inference features)
        lag_features = [f for f in available_features if 'lag_' in f]
        if lag_features:
            feature_sets['lag_features'] = {
                'features': lag_features,
                'description': 'Lag features (causal inference)'
            }
            print(f"   • Found {len(lag_features)} lag features (causal inference)")
        else:
            print("     No lag features found - causal inference may not have been applied")

        # 3. Non-lag features
        non_lag_features = [f for f in available_features if 'lag_' not in f]
        if non_lag_features:
            feature_sets['non_lag_features'] = {
                'features': non_lag_features,
                'description': 'Non-lag features (traditional)'
            }

        # 4. Economic indicators (if available)
        economic_features = [f for f in available_features if any(econ in f.lower() for econ in 
                           ['unemployment', 'cpi', 'rate', 'gdp', 'inflation', 'yield'])]
        if economic_features:
            feature_sets['economic_features'] = {
                'features': economic_features,
                'description': 'Economic indicators'
            }

        # 5. Technical indicators (if available)
        technical_features = [f for f in available_features if any(tech in f.lower() for tech in 
                            ['rsi', 'macd', 'sma', 'ema', 'bollinger', 'volume', 'return'])]
        if technical_features:
            feature_sets['technical_features'] = {
                'features': technical_features,
                'description': 'Technical indicators'
            }

        # Display feature sets
        print(f"\\nFeature Set Summary:")
        for name, fset in feature_sets.items():
            print(f"   • {name}: {len(fset['features'])} features - {fset['description']}")

        self.feature_sets = feature_sets
        return feature_sets

    def create_datasets(self):
        """Create datasets """
        print(f"\\n CREATING DATASETS FROM data")
        print("-" * 50)

        # Ensure Date column is datetime if it exists
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            # Sort by date for time series split
            if 'Ticker' in self.data.columns:
                self.data = self.data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
            else:
                self.data = self.data.sort_values('Date').reset_index(drop=True)

        # Prepare datasets
        datasets = {}

        for set_name, feature_info in self.feature_sets.items():
            features = feature_info['features']

            # Check which features actually exist
            existing_features = [f for f in features if f in self.data.columns]

            if len(existing_features) == 0:
                print(f"   Skipping {set_name} - no existing features")
                continue

            # Create dataset
            X = self.data[existing_features].copy()

            # Separate numeric and categorical columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns

            # Handle missing values separately for numeric and categorical
            if len(numeric_cols) > 0:
                missing_counts = X[numeric_cols].isnull().sum()
                if missing_counts.sum() > 0:
                    print(f"    {set_name}: Found missing values in {(missing_counts > 0).sum()} numeric features")
                    # Handle missing values with median imputation for numeric columns
                    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

            if len(categorical_cols) > 0:
                missing_counts_cat = X[categorical_cols].isnull().sum()
                if missing_counts_cat.sum() > 0:
                    print(f"    {set_name}: Found missing values in {(missing_counts_cat > 0).sum()} categorical features")
                    # Handle missing values with mode imputation for categorical columns
                    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

            # Handle categorical columns - convert to numeric using label encoding
            if len(categorical_cols) > 0:
                print(f"    {set_name}: Converting {len(categorical_cols)} categorical features to numeric")
                from sklearn.preprocessing import LabelEncoder

                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # Check for infinite values in numeric columns
            if len(numeric_cols) > 0:
                inf_counts = np.isinf(X[numeric_cols]).sum()
                if inf_counts.sum() > 0:
                    print(f"    {set_name}: Found infinite values, replacing with median")
                    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(X[numeric_cols].median())

            datasets[set_name] = {
                'X': X,
                'features': existing_features,
                'feature_count': len(existing_features),
                'description': feature_info['description']
            }

            print(f"   {set_name}: {len(existing_features)} features ready")

        if len(datasets) == 0:
            raise ValueError("No valid datasets created. Check your feature columns.")

        # Target variable
        y = self.data[self.target].copy()

        print(f"\\nFinal Dataset Summary:")
        for name, dataset in datasets.items():
            print(f"   • {name}: {dataset['feature_count']} features")
        print(f"   • Target samples: {len(y)}")
        print(f"   • Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

        # Check for class imbalance
        class_counts = y.value_counts()
        if len(class_counts) > 1:
            minority_ratio = class_counts.min() / class_counts.max()
            if minority_ratio < 0.1:
                print(f"    Severe class imbalance detected (ratio: {minority_ratio:.3f})")

        self.datasets = datasets
        self.y = y

        return datasets

    def train_and_evaluate_models(self):
        """Train and evaluate models on data"""
        print(f"\\n TRAINING AND EVALUATING MODELS ON data")
        print("-" * 50)

        # Define models
        models_config = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'name': 'Logistic Regression',
                'scale_features': True
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'name': 'Random Forest',
                'scale_features': False
            },
            'xgboost':{'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
                'name': 'XGBoost',
                'scale_features': False
            }
        }

       
        results = []

        # Train each model on each dataset
        for dataset_name, dataset in self.datasets.items():
            print(f"\\nProcessing {dataset_name} ({dataset['feature_count']} features)...")

            X = dataset['X']

            # Check if we have enough data for train/test split
            if len(X) < 50:
                print(f"     Insufficient data ({len(X)} samples) for reliable evaluation")
                continue

            # Create train/test split (80/20) but ensure minimum test size
            test_size = max(int(len(X) * 0.2), 10)
            split_idx = len(X) - test_size
            
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = self.y.iloc[:split_idx], self.y.iloc[split_idx:]

            print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

            # Check class distribution in train/test
            train_dist = y_train.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)
            print(f"   Train class distribution: {train_dist.round(3).to_dict()}")
            print(f"   Test class distribution: {test_dist.round(3).to_dict()}")

            for model_key, model_config in models_config.items():
                print(f"    Training {model_config['name']}...")

                try:
                    # Prepare data
                    X_train_processed = X_train.copy()
                    X_test_processed = X_test.copy()

                    if model_config['scale_features']:
                        scaler = StandardScaler()
                        X_train_processed = pd.DataFrame(
                            scaler.fit_transform(X_train_processed),
                            columns=X_train_processed.columns,
                            index=X_train_processed.index
                        )
                        X_test_processed = pd.DataFrame(
                            scaler.transform(X_test_processed),
                            columns=X_test_processed.columns,
                            index=X_test_processed.index
                        )

                    # Train model
                    model = model_config['model']
                    model.fit(X_train_processed, y_train)

                    # Predictions
                    y_pred = model.predict(X_test_processed)
                    y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    # Calculate feature importance if available
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X_train_processed.columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        feature_importance = dict(zip(X_train_processed.columns, abs(model.coef_[0])))

                    # Simple trading simulation (if we have return data)
                    strategy_returns = []
                    if 'Daily_Return' in self.data.columns:
                        # Use actual returns for simulation
                        actual_returns = self.data['Daily_Return'].iloc[split_idx:split_idx+len(y_test)].values
                        for i in range(len(y_pred)):
                            if i < len(actual_returns):
                                if y_pred[i] == 1:  # Predicted up
                                    strategy_returns.append(actual_returns[i])
                                else:
                                    strategy_returns.append(0)  # No position
                    else:
                        # Fallback to simple simulation
                        for i in range(len(y_pred)):
                            if y_pred[i] == 1:  # Predicted up
                                actual_return = 0.01 if y_test.iloc[i] == 1 else -0.01
                            else:
                                actual_return = 0
                            strategy_returns.append(actual_return)

                    strategy_returns = np.array(strategy_returns)
                    total_return = np.sum(strategy_returns)
                    volatility = np.std(strategy_returns) if len(strategy_returns) > 1 else 0
                    sharpe_ratio = (np.mean(strategy_returns) / volatility * np.sqrt(252)) if volatility > 0 else 0

                    # Store results
                    result = {
                        'Dataset': dataset_name,
                        'Model': model_config['name'],
                        'Features': dataset['feature_count'],
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1_Score': f1,
                        'Total_Return': total_return,
                        'Sharpe_Ratio': sharpe_ratio,
                        'Description': dataset['description'],
                        'Train_Size': len(X_train),
                        'Test_Size': len(X_test)
                    }

                    # Add top features if available
                    if feature_importance:
                        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        result['Top_Features'] = [f[0] for f in top_features]

                    results.append(result)

                    print(f"      Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Sharpe: {sharpe_ratio:.4f}")

                except Exception as e:
                    print(f"      Error with {model_config['name']}: {str(e)[:100]}...")
                    continue

        if len(results) == 0:
            raise ValueError("No models were successfully trained. Check your data and features.")

        # Create results DataFrame
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def create_comparison_report(self):
        """Create comprehensive comparison report for data results"""
        print(f"\\n CREATING COMPARISON REPORT")
        print("-" * 50)

        if self.results_df is None or self.results_df.empty:
            print("No results to report")
            return None

        # Display results
        print(f"\\n MODEL COMPARISON RESULTS (data)")
        print("=" * 80)

        # Sort by F1 score (more balanced than accuracy for potentially imbalanced data)
        results_sorted = self.results_df.sort_values('F1_Score', ascending=False)

        # Display results
        display_cols = ['Dataset', 'Model', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'Sharpe_Ratio']
        available_cols = [col for col in display_cols if col in results_sorted.columns]
        print(results_sorted[available_cols].round(4).to_string(index=False))

        # Key insights
        print(f"\\nKEY INSIGHTS FROM data:")

        if len(results_sorted) > 0:
            best_model = results_sorted.iloc[0]
            print(f"   Best Model: {best_model['Model']} on {best_model['Dataset']}")
            print(f"      • F1 Score: {best_model['F1_Score']:.4f}")
            print(f"      • Accuracy: {best_model['Accuracy']:.4f}")
            print(f"      • Sharpe Ratio: {best_model['Sharpe_Ratio']:.4f}")
            print(f"      • Features: {best_model['Features']}")

            # Show top features if available
            if 'Top_Features' in best_model and best_model['Top_Features']:
                print(f"      • Top Features: {', '.join(best_model['Top_Features'][:3])}")

        # Feature efficiency analysis
        if len(results_sorted) > 1:
            print(f"\\n   Feature Efficiency Analysis:")
            min_features = self.results_df['Features'].min()
            max_features = self.results_df['Features'].max()
            print(f"      • Feature range: {min_features} to {max_features}")

            # Compare lag vs non-lag if available
            lag_results = self.results_df[self.results_df['Dataset'] == 'lag_features']
            all_results = self.results_df[self.results_df['Dataset'] == 'all_features']

            if not lag_results.empty and not all_results.empty:
                lag_avg_f1 = lag_results['F1_Score'].mean()
                all_avg_f1 = all_results['F1_Score'].mean()
                print(f"      • Lag features (causal) avg F1: {lag_avg_f1:.4f}")
                print(f"      • All features avg F1: {all_avg_f1:.4f}")
                if all_avg_f1 > 0:
                    print(f"      • Causal vs All performance: {((lag_avg_f1 - all_avg_f1) / all_avg_f1 * 100):+.2f}%")

            # Model comparison
            model_performance = self.results_df.groupby('Model')['F1_Score'].mean().sort_values(ascending=False)
            print(f"\\n  Model Performance Ranking:")
            for model, score in model_performance.items():
                print(f"      • {model}: {score:.4f}")

        # Data quality insights
        print(f"\\n   Data Quality Insights:")
        total_samples = len(self.data)
        print(f"      • Total samples used: {total_samples}")
        print(f"      • Features available: {len([col for col in self.data.columns if col not in ['Date', 'Ticker', 'Target_Direction']])}")
        
        if 'Train_Size' in self.results_df.columns:
            avg_train_size = self.results_df['Train_Size'].mean()
            avg_test_size = self.results_df['Test_Size'].mean()
            print(f"      • Average train/test split: {avg_train_size:.0f}/{avg_test_size:.0f}")

        return self.results_df

    def run_complete_analysis(self):
        """Run the complete analysis pipeline on data"""
        print("STARTING data PREDICTIVE MODELING ANALYSIS")
        print("=" * 80)

        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()

            # Step 2: Prepare feature sets
            self.prepare_feature_sets()

            # Step 3: Create datasets
            self.create_datasets()

            # Step 4: Train and evaluate models
            self.train_and_evaluate_models()

            # Step 5: Create comparison report
            self.create_comparison_report()

            print(f"\\ndata ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"Results available in pipeline.results_df")

            return self.results_df

        except Exception as e:
            print(f"Error in data analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

# Convenience function for data only
def run_real_data_modeling_analysis(df):
    """
    Run complete modeling analysis on data only

    Parameters:
    -----------
    df : pd.DataFrame
        Your processed dataframe with features and target from causal inference step
        Must contain 'Target_Direction' column and feature columns

    Returns:
    --------
    pipeline : RealDataPredictiveModelingPipeline
        The pipeline object with all results
    results : pd.DataFrame
        Results comparison table
    """
    if df is None:
        raise ValueError("DataFrame is required. This pipeline only works with data from causal inference.")
    
    pipeline = RealDataPredictiveModelingPipeline(df=df)
    results = pipeline.run_complete_analysis()
    return pipeline, results

# Main execution
if __name__ == "__main__":
    print("🇬🇧 data ONLY PREDICTIVE MODELING ANALYSIS")
    print("=" * 70)
    print("This script requires data from the causal inference step")
    print("Usage: pipeline, results = run_real_data_modeling_analysis(your_df)")
    print("=" * 70)
    
    print("Cannot run without data. Please provide your dataframe:")
    print("   pipeline, results = run_real_data_modeling_analysis(df=your_dataframe)")

print("data only modeling script ready!")
print("\\nTo run with your data:")
print("pipeline, results = run_real_data_modeling_analysis(df=your_dataframe)")
