import pandas as pd
import numpy as np
import json
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import os
warnings.filterwarnings('ignore')

# Ensure reproducibility
def ensure_reproducible_results():
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'

ensure_reproducible_results()


class RollingWindowNormalizer:
    """Rolling Window Normalization for time series data"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fitted = False
    
    def fit_transform(self, X):
        """Fit and transform the data using rolling window normalization"""
        X_normalized = X.copy()
        
        for col in X.columns:
            # Calculate rolling mean and std
            rolling_mean = X[col].rolling(window=self.window_size, min_periods=1).mean()
            rolling_std = X[col].rolling(window=self.window_size, min_periods=1).std()
            
            # Normalize: (value - rolling_mean) / rolling_std
            # Handle division by zero by replacing with 0
            X_normalized[col] = (X[col] - rolling_mean) / rolling_std.replace(0, 1)
        
        self.fitted = True
        return X_normalized
    
    def transform(self, X):
        """Transform new data (for test set, use last window statistics)"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        # For test data, we'll use expanding window from the beginning
        return self.fit_transform(X)


class RealDataPredictiveModelingPipeline:
    """
    Predictive modeling pipeline from causal inference step with causal metadata integration
    """

    def __init__(self, df, causal_metadata_path=None):
        if df is None:
            raise ValueError("DataFrame is required.")
        self.df = df
        self.causal_metadata_path = causal_metadata_path
        self.causal_features = []
        self.results_df = None
        self.train_test_metrics = []  # Store detailed train/test metrics
        self.load_causal_metadata()
        
        self.plots_dir = 'C:/Users/sheri/Downloads/final project/Causal_Inference/output'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            

    def load_causal_metadata(self):
        """Load causal features from metadata JSON file"""
        if self.causal_metadata_path:
            try:
                with open(self.causal_metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Extract distinct causal features
                raw_causal_features = metadata.get('causal_features', [])
                self.causal_features = list(set(raw_causal_features))  # Remove duplicates

                print(f"Loaded {len(self.causal_features)} distinct causal features from metadata")
                print(f"Causal features: {self.causal_features[:5]}{'...' if len(self.causal_features) > 5 else ''}")

            except Exception as e:
                print(f"Warning: Could not load causal metadata: {e}")
                self.causal_features = []
        else:
            print("No causal metadata path provided")

    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        print("LOADING AND PREPARING data")
        print("-" * 50)

        self.data = self.df.copy()
        print(f"Using provided dataframe: {len(self.data)} records")

        # Prepare target variable
        self.target = 'Target_Direction'

        # Data quality check
        print(f"\n Data Quality Check:")
        print(f" • Dataset shape: {self.data.shape}")
        print(f" • Available columns: {list(self.data.columns)}")

        if self.target in self.data.columns:
            print(f" • Target distribution: {self.data[self.target].value_counts().to_dict()}")
            print(f" • Missing values in target: {self.data[self.target].isna().sum()}")

            # Remove rows with missing target
            initial_len = len(self.data)
            self.data = self.data.dropna(subset=[self.target])
            final_len = len(self.data)

            if initial_len != final_len:
                print(f" • Removed {initial_len - final_len} rows with missing target")
        else:
            raise ValueError(f"Target column '{self.target}' not found in the data. Available columns: {list(self.data.columns)}")

        # Check if we have enough data
        if len(self.data) < 100:
            print(f"  Warning: Only {len(self.data)} samples available. Results may not be reliable.")

        return self.data

    def prepare_feature_sets(self):
        """Prepare different feature sets including causal features from metadata"""
        print(f"\n PREPARING FEATURE SETS FROM data")
        print("-" * 50)

        # Exclude non-feature columns
        exclude_cols = ['Date', 'Ticker', 'Target_Direction', 'Target_Excess_Return',
                       'Target_5D_Return', 'Target_10D_Return']

        # Get all available features
        available_features = [col for col in self.data.columns if col not in exclude_cols]

        if len(available_features) == 0:
            raise ValueError("No features found in the data after excluding target and metadata columns.")

        print(f"Available features: {len(available_features)}")
        print(f" Features: {available_features[:10]}{'...' if len(available_features) > 10 else ''}")

        # Create feature sets
        feature_sets = {}

        # 1. All features
        feature_sets['all_features'] = {
            'features': available_features,
            'description': 'All available features'
        }

        # 2. Causal features from metadata (if available)
        if self.causal_features:
            # Filter causal features that exist in the data
            existing_causal_features = [f for f in self.causal_features if f in available_features]
            if existing_causal_features:
                feature_sets['causal_features'] = {
                    'features': existing_causal_features,
                    'description': 'Causal features from metadata'
                }
                print(f" • Found {len(existing_causal_features)} causal features from metadata")
            else:
                print("   • No causal features from metadata found in data")

        # Display feature sets
        print(f"\nFeature Set Summary:")
        for name, fset in feature_sets.items():
            print(f" • {name}: {len(fset['features'])} features - {fset['description']}")

        self.feature_sets = feature_sets
        return feature_sets

    def create_datasets(self):
        """Create datasets"""
        print(f"\n CREATING DATASETS FROM data")
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
                print(f" Skipping {set_name} - no existing features")
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
                    print(f" {set_name}: Found missing values in {(missing_counts > 0).sum()} numeric features")
                    # Handle missing values with median imputation for numeric columns
                    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

            if len(categorical_cols) > 0:
                missing_counts_cat = X[categorical_cols].isnull().sum()
                if missing_counts_cat.sum() > 0:
                    print(f" {set_name}: Found missing values in {(missing_counts_cat > 0).sum()} categorical features")
                    # Handle missing values with mode imputation for categorical columns
                    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

            # Handle categorical columns - convert to numeric using label encoding
            if len(categorical_cols) > 0:
                print(f" {set_name}: Converting {len(categorical_cols)} categorical features to numeric")
                from sklearn.preprocessing import LabelEncoder

                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # Check for infinite values in numeric columns
            if len(numeric_cols) > 0:
                inf_counts = np.isinf(X[numeric_cols]).sum()
                if inf_counts.sum() > 0:
                    print(f" {set_name}: Found infinite values, replacing with median")
                    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(X[numeric_cols].median())

            datasets[set_name] = {
                'X': X,
                'features': existing_features,
                'feature_count': len(existing_features),
                'description': feature_info['description']
            }

            print(f" {set_name}: {len(existing_features)} features ready")

        if len(datasets) == 0:
            raise ValueError("No valid datasets created. Check your feature columns.")

        # Target variable
        y = self.data[self.target].copy()

        print(f"\nFinal Dataset Summary:")
        for name, dataset in datasets.items():
            print(f" • {name}: {dataset['feature_count']} features")
            print(f" • Target samples: {len(y)}")
            print(f" • Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

        # Check for class imbalance
        class_counts = y.value_counts()
        if len(class_counts) > 1:
            minority_ratio = class_counts.min() / class_counts.max()
            if minority_ratio < 0.1:
                print(f" Severe class imbalance detected (ratio: {minority_ratio:.3f})")

        self.datasets = datasets
        self.y = y

        return datasets

    def train_and_evaluate_models(self):
        """Train and evaluate models on data"""
        print(f"\n TRAINING AND EVALUATING MODELS ON data")
        print("-" * 50)

        # Define models
        
        #RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(
                                        n_estimators=100,#100  
                                        max_depth=10, #10      
                                        min_samples_split=20,#20  
                                        min_samples_leaf=10,#10   
                                        max_features=0.3,      
                                        random_state=42,
                                        n_jobs=1),
                'name': 'Random Forest',
                'scale_features': False
            }
        }

        results = []
        self.train_test_metrics = []  # Reset metrics storage

        # Train each model on each dataset
        for dataset_name, dataset in self.datasets.items():
            print(f"\nProcessing {dataset_name} ({dataset['feature_count']} features)...")

            X = dataset['X']

            # Check if we have enough data for train/test split
            if len(X) < 50:
                print(f" Insufficient data ({len(X)} samples) for reliable evaluation")
                continue

            # Create train/test split (80/20) but ensure minimum test size
            test_size = max(int(len(X) * 0.2), 10)
            split_idx = len(X) - test_size

            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = self.y.iloc[:split_idx], self.y.iloc[split_idx:]

            print(f" Train: {len(X_train)}, Test: {len(X_test)}")

            # Check class distribution in train/test
            train_dist = y_train.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)
            print(f" Train class distribution: {train_dist.round(3).to_dict()}")
            print(f" Test class distribution: {test_dist.round(3).to_dict()}")

            for model_key, model_config in models_config.items():
                print(f" Training {model_config['name']}...")

                try:
                    # Prepare data
                    X_train_processed = X_train.copy()
                    X_test_processed = X_test.copy()

                    if model_config['scale_features']:
                        normalizer = RollingWindowNormalizer(window_size=30)
                        X_train_processed = normalizer.fit_transform(X_train_processed)
                        X_test_processed = normalizer.transform(X_test_processed)

                    # Train model
                    model = model_config['model']
                    model.fit(X_train_processed, y_train)

                    # Predictions on both train and test sets
                    y_train_pred = model.predict(X_train_processed)
                    y_test_pred = model.predict(X_test_processed)
                    
                    y_test_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None

                    # Calculate train metrics
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
                    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
                    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

                    # Calculate test metrics
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

                    # Calculate feature importance if available
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X_train_processed.columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        feature_importance = dict(zip(X_train_processed.columns, abs(model.coef_[0])))

                    # Simple trading simulation
                    strategy_returns = []
                    if 'Daily_Return' in self.data.columns:
                        # Use actual returns for simulation
                        actual_returns = self.data['Daily_Return'].iloc[split_idx:split_idx+len(y_test)].values
                        for i in range(len(y_test_pred)):
                            if i < len(actual_returns):
                                if y_test_pred[i] == 1:  # Predicted up
                                    strategy_returns.append(actual_returns[i])
                                else:
                                    strategy_returns.append(0)  # No position
                    else:
                        # Fallback to simple simulation
                        for i in range(len(y_test_pred)):
                            if y_test_pred[i] == 1:  # Predicted up
                                actual_return = 0.01 if y_test.iloc[i] == 1 else -0.01
                            else:
                                actual_return = 0
                            strategy_returns.append(actual_return)

                    strategy_returns = np.array(strategy_returns)
                    total_return = np.sum(strategy_returns)
                    volatility = np.std(strategy_returns) if len(strategy_returns) > 1 else 0
                    sharpe_ratio = (np.mean(strategy_returns) / volatility * np.sqrt(252)) if volatility > 0 else 0

                    # Store detailed train/test metrics
                    train_test_metric = {
                        'Dataset': dataset_name,
                        'Model': model_config['name'],
                        'Split': 'Train',
                        'Accuracy': train_accuracy,
                        'Precision': train_precision,
                        'Recall': train_recall,
                        'F1_Score': train_f1,
                        'Size': len(X_train)
                    }
                    self.train_test_metrics.append(train_test_metric)

                    train_test_metric = {
                        'Dataset': dataset_name,
                        'Model': model_config['name'],
                        'Split': 'Test',
                        'Accuracy': test_accuracy,
                        'Precision': test_precision,
                        'Recall': test_recall,
                        'F1_Score': test_f1,
                        'Size': len(X_test)
                    }
                    self.train_test_metrics.append(train_test_metric)

                    # Store main results (using test metrics)
                    result = {
                        'Dataset': dataset_name,
                        'Model': model_config['name'],
                        'Features': dataset['feature_count'],
                        'Train_Accuracy': train_accuracy,
                        'Test_Accuracy': test_accuracy,
                        'Train_F1': train_f1,
                        'Test_F1': test_f1,
                        'Accuracy': test_accuracy,  # Keep for backward compatibility
                        'Precision': test_precision,
                        'Recall': test_recall,
                        'F1_Score': test_f1,
                        'Total_Return': total_return,
                        'Sharpe_Ratio': sharpe_ratio,
                        'Description': dataset['description'],
                        'Train_Size': len(X_train),
                        'Test_Size': len(X_test),
                        'Overfitting_Score': train_accuracy - test_accuracy  # Simple overfitting indicator
                    }

                    # Add top features if available
                    if feature_importance:
                        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        result['Top_Features'] = [f[0] for f in top_features]

                    results.append(result)

                    print(f" Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
                    print(f" Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Sharpe: {sharpe_ratio:.4f}")
                    print(f" Overfitting Score: {train_accuracy - test_accuracy:.4f}")

                except Exception as e:
                    print(f" Error with {model_config['name']}: {str(e)[:100]}...")
                    continue

        if len(results) == 0:
            raise ValueError("No models were successfully trained. Check your data and features.")

        # Create results DataFrame
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def plot_accuracy_comparison(self):
        """Create accuracy comparison plots"""
        if self.train_test_metrics is None or len(self.train_test_metrics) == 0:
            print("No train/test metrics available for plotting")
            return

        # Convert to DataFrame for easier plotting
        metrics_df = pd.DataFrame(self.train_test_metrics)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis: Train vs Test Metrics', fontsize=16, fontweight='bold')

        # 1. Accuracy Comparison (Bar Plot)
        ax1 = axes[0, 0]
        accuracy_pivot = metrics_df.pivot_table(values='Accuracy', index=['Dataset', 'Model'], columns='Split')
        accuracy_pivot.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title('Accuracy: Train vs Test', fontweight='bold')
        ax1.set_ylabel('Accuracy Score')
        ax1.legend(title='Split')
        ax1.tick_params(axis='x', rotation=45)

        # 2. F1 Score Comparison (Bar Plot)
        ax2 = axes[0, 1]
        f1_pivot = metrics_df.pivot_table(values='F1_Score', index=['Dataset', 'Model'], columns='Split')
        f1_pivot.plot(kind='bar', ax=ax2, color=['lightgreen', 'orange'])
        ax2.set_title('F1 Score: Train vs Test', fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.legend(title='Split')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Overfitting Analysis (if results_df is available)
        ax3 = axes[1, 0]
        if self.results_df is not None and 'Overfitting_Score' in self.results_df.columns:
            overfitting_data = self.results_df[['Dataset', 'Model', 'Overfitting_Score']].copy()
            overfitting_data['Model_Dataset'] = overfitting_data['Model'] + ' - ' + overfitting_data['Dataset']
            
            bars = ax3.bar(range(len(overfitting_data)), overfitting_data['Overfitting_Score'], 
                          color=['red' if x > 0.05 else 'green' for x in overfitting_data['Overfitting_Score']])
            ax3.set_title('Overfitting Analysis (Train Acc - Test Acc)', fontweight='bold')
            ax3.set_ylabel('Overfitting Score')
            ax3.set_xlabel('Model - Dataset')
            ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_xticks(range(len(overfitting_data)))
            ax3.set_xticklabels(overfitting_data['Model_Dataset'], rotation=45, ha='right')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Overfitting data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Overfitting Analysis', fontweight='bold')

        # 4. Performance Summary (if results_df is available)
        ax4 = axes[1, 1]
        if self.results_df is not None:
            # Create a summary scatter plot
            scatter_data = self.results_df[['Train_Accuracy', 'Test_Accuracy', 'Dataset', 'Model']].copy()
            colors = ['blue' if 'causal' in dataset.lower() else 'red' for dataset in scatter_data['Dataset']]
            
            scatter = ax4.scatter(scatter_data['Train_Accuracy'], scatter_data['Test_Accuracy'], 
                                c=colors, alpha=0.7, s=100)
            
            # Add diagonal line (perfect generalization)
            min_acc = min(scatter_data['Train_Accuracy'].min(), scatter_data['Test_Accuracy'].min())
            max_acc = max(scatter_data['Train_Accuracy'].max(), scatter_data['Test_Accuracy'].max())
            ax4.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5, label='Perfect Generalization')
            
            ax4.set_xlabel('Train Accuracy')
            ax4.set_ylabel('Test Accuracy')
            ax4.set_title('Generalization Analysis', fontweight='bold')
            ax4.legend(['Perfect Generalization', 'Causal Features', 'All Features'])
            
            # Add annotations for each point
            for i, row in scatter_data.iterrows():
                ax4.annotate(f"{row['Model'][:2]}-{row['Dataset'][:3]}", 
                           (row['Train_Accuracy'], row['Test_Accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Results data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Generalization Analysis', fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'Performance_Metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        #plt.show()

    def create_comparison_report(self):
        """Create comprehensive comparison report with train/test metrics"""
        print(f"\n CREATING COMPARISON REPORT")
        print("-" * 50)

        if self.results_df is None or self.results_df.empty:
            print("No results to report")
            return None

        # Display results with train/test metrics
        print(f"\n MODEL COMPARISON RESULTS (with Train/Test Metrics)")
        print("=" * 100)

        # Sort by Test F1 score
        results_sorted = self.results_df.sort_values('Test_F1', ascending=False)

        # Display results with train/test columns
        display_cols = ['Dataset', 'Model', 'Features', 'Train_Accuracy', 'Test_Accuracy', 
                       'Train_F1', 'Test_F1', 'Overfitting_Score', 'Sharpe_Ratio']
        available_cols = [col for col in display_cols if col in results_sorted.columns]
        print(results_sorted[available_cols].round(4).to_string(index=False))

        # Train/Test Metrics Summary
        if self.train_test_metrics:
            print(f"\n DETAILED TRAIN/TEST METRICS SUMMARY")
            print("=" * 80)
            metrics_df = pd.DataFrame(self.train_test_metrics)
            
            # Group by Dataset and Model, show train vs test
            for (dataset, model), group in metrics_df.groupby(['Dataset', 'Model']):
                print(f"\n{model} on {dataset}:")
                train_metrics = group[group['Split'] == 'Train'].iloc[0]
                test_metrics = group[group['Split'] == 'Test'].iloc[0]
                
                print(f"  Train: Acc={train_metrics['Accuracy']:.4f}, F1={train_metrics['F1_Score']:.4f}, Size={train_metrics['Size']}")
                print(f"  Test:  Acc={test_metrics['Accuracy']:.4f}, F1={test_metrics['F1_Score']:.4f}, Size={test_metrics['Size']}")
                print(f"  Gap:   Acc={train_metrics['Accuracy'] - test_metrics['Accuracy']:.4f}, F1={train_metrics['F1_Score'] - test_metrics['F1_Score']:.4f}")

        # Key insights
        print(f"\nKEY INSIGHTS:")

        if len(results_sorted) > 0:
            best_model = results_sorted.iloc[0]
            print(f" Best Model: {best_model['Model']} on {best_model['Dataset']}")
            print(f" • Train Accuracy: {best_model['Train_Accuracy']:.4f}")
            print(f" • Test Accuracy: {best_model['Test_Accuracy']:.4f}")
            print(f" • Train F1: {best_model['Train_F1']:.4f}")
            print(f" • Test F1: {best_model['Test_F1']:.4f}")
            print(f" • Overfitting Score: {best_model['Overfitting_Score']:.4f}")
            print(f" • Sharpe Ratio: {best_model['Sharpe_Ratio']:.4f}")
            print(f" • Features: {best_model['Features']}")

            # Show top features if available
            if 'Top_Features' in best_model and best_model['Top_Features']:
                print(f" • Top Features: {', '.join(best_model['Top_Features'][:3])}")

            # Overfitting analysis
            if best_model['Overfitting_Score'] > 0.05:
                print(f" Warning: Potential overfitting detected (gap > 0.05)")
            else:
                print(f"  Good generalization (low train-test gap)")

        # Compare causal vs all features if available
        causal_results = self.results_df[self.results_df['Dataset'] == 'causal_features']
        all_results = self.results_df[self.results_df['Dataset'] == 'all_features']

        if not causal_results.empty and not all_results.empty:
            causal_avg_test_f1 = causal_results['Test_F1'].mean()
            all_avg_test_f1 = all_results['Test_F1'].mean()
            causal_avg_overfitting = causal_results['Overfitting_Score'].mean()
            all_avg_overfitting = all_results['Overfitting_Score'].mean()
            
            print(f"\n   Causal vs All Features Comparison:")
            print(f" • Causal features avg Test F1: {causal_avg_test_f1:.4f}")
            print(f" • All features avg Test F1: {all_avg_test_f1:.4f}")
            print(f" • Causal features avg Overfitting: {causal_avg_overfitting:.4f}")
            print(f" • All features avg Overfitting: {all_avg_overfitting:.4f}")
            
            if all_avg_test_f1 > 0:
                print(f" • Test F1 Performance difference: {((causal_avg_test_f1 - all_avg_test_f1) / all_avg_test_f1 * 100):+.2f}%")
            
            if causal_avg_overfitting < all_avg_overfitting:
                print(f"  Causal features show better generalization")
            else:
                print(f" All features show better generalization")

        # Generate accuracy plot
        print(f"\n GENERATING ACCURACY PLOTS...")
        self.plot_accuracy_comparison()

        return self.results_df

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("STARTING PREDICTIVE MODELING ANALYSIS WITH CAUSAL FEATURES")
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

            print(f"\nANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"Results available in pipeline.results_df")
            print(f"Train/Test metrics available in pipeline.train_test_metrics")

            return self.results_df

        except Exception as e:
            print(f"Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            return None


# Convenience function
def run_causal_modeling_analysis(modelling_df, causal_metadata_path=None):
    """
    Run complete modeling analysis with causal features

    Parameters:
    ----
    modelling_df : pd.DataFrame
        Your processed dataframe with features and target
    causal_metadata_path : str, optional
        Path to causal analysis metadata JSON file

    Returns:
    ----
    pipeline : RealDataPredictiveModelingPipeline
        The pipeline object with all results
    results : pd.DataFrame
        Results comparison table
    """
    if modelling_df is None:
        raise ValueError("modelling_df is required.")

    pipeline = RealDataPredictiveModelingPipeline(df=modelling_df, causal_metadata_path=causal_metadata_path)
    results = pipeline.run_complete_analysis()
    return pipeline, results


# Main execution
if __name__ == "__main__":
    print("CAUSAL FEATURES PREDICTIVE MODELING ANALYSIS")
    print("=" * 70)
    print("This script integrates causal features from metadata with all features")
    print("Now includes detailed train/test metrics and accuracy plots")
    print("Usage: pipeline, results = run_causal_modeling_analysis(modelling_df, 'causal_analysis_metadata.json')")
    print("=" * 70)

    print("To run with your data:")
    print("pipeline, results = run_causal_modeling_analysis(modelling_df=your_dataframe, causal_metadata_path='causal_analysis_metadata.json')")

print("Enhanced modeling script with train/test metrics and plots ready!")