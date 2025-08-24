import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Setup logging
def setup_logging():
    log_dir = Path('Log')
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'stock_prediction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Set random seeds
np.random.seed(42)

class MetadataLoader:
    """Handles loading and validation of metadata files"""

    @staticmethod
    def load_metadata(filepath):
        """Load metadata from JSON file"""
        try:
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata from {filepath}")
            logger.info(f"Loaded metadata from {filepath}")
            return metadata
        except FileNotFoundError:
            print(f"ERROR: Metadata file not found: {filepath}")
            logger.error(f"Metadata file not found: {filepath}")
            return {}
        except json.JSONDecodeError:
            print(f"ERROR: Invalid JSON in metadata file: {filepath}")
            logger.error(f"Invalid JSON in metadata file: {filepath}")
            return {}

    @staticmethod
    def extract_features(metadata):
        """Extract feature list from metadata"""
        if 'selected_features' in metadata:
            return metadata['selected_features']
        elif 'causal_features' in metadata:
            return metadata['causal_features']
        else:
            print("WARNING: No recognized feature key in metadata")
            logger.warning("No recognized feature key in metadata")
            return []

class DataProcessor:
    """Handles data preprocessing and feature set preparation"""

    def __init__(self, df):
        self.df = df.copy()
        self.target = 'Target_Direction'
        self.exclude_cols = ['Date', 'Ticker', 'Target_Direction',
                           'Target_Excess_Return', 'Target_5D_Return', 'Target_10D_Return']

    def prepare_data(self):
        """Clean and prepare data for modeling"""
        print("Preparing data for modeling")
        logger.info("Preparing data for modeling")

        # Remove rows with missing target
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=[self.target])
        final_len = len(self.df)

        if initial_len != final_len:
            print(f"Removed {initial_len - final_len} rows with missing target")
            logger.info(f"Removed {initial_len - final_len} rows with missing target")

        # Sort by date if available
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            if 'Ticker' in self.df.columns:
                self.df = self.df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
            else:
                self.df = self.df.sort_values('Date').reset_index(drop=True)

        print(f"Data prepared: {len(self.df)} records")
        logger.info(f"Data prepared: {len(self.df)} records")
        return self.df

    def create_feature_sets(self, selected_features, causal_features):
        """Create feature sets for comparison"""
        available_features = [col for col in self.df.columns if col not in self.exclude_cols]

        # Filter features that exist in data
        selected_features = [f for f in selected_features if f in available_features]
        causal_features = [f for f in causal_features if f in available_features]

        feature_sets = {
            'selected_features': {
                'features': selected_features,
                'description': 'Selected features from feature selection',
                'count': len(selected_features)
            },
            'causal_features': {
                'features': causal_features,
                'description': 'Causal features from causal inference',
                'count': len(causal_features)
            }
        }

        print(f"Selected features: {len(selected_features)}")
        print(f"Causal features: {len(causal_features)}")
        logger.info(f"Selected features: {len(selected_features)}")
        logger.info(f"Causal features: {len(causal_features)}")

        return feature_sets

    def create_datasets(self, feature_sets):
        """Create X, y datasets for each feature set"""
        datasets = {}
        y = self.df[self.target].copy()

        for set_name, feature_info in feature_sets.items():
            features = feature_info['features']

            if not features:
                print(f"WARNING: No features available for {set_name}")
                logger.warning(f"No features available for {set_name}")
                continue

            X = self.df[features].copy()

            # Handle missing values
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

            # Handle categorical columns
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(categorical_cols) > 0:
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            datasets[set_name] = {
                'X': X,
                'y': y,
                'features': features,
                'feature_count': len(features),
                'description': feature_info['description']
            }

        return datasets

class ModelTrainer:
    """Handles model training and hyperparameter tuning"""

    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbosity=-1)
        }

        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [10, 20],
                'max_features': [0.3, 0.5, 'sqrt']
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'num_leaves': [31, 50]
            }
        }

    def tune_hyperparameters(self, X_train, y_train, model_name):
        """Tune hyperparameters for a specific model"""
        print(f"Tuning hyperparameters for {model_name}")
        logger.info(f"Tuning hyperparameters for {model_name}")

        base_model = self.models[model_name]
        param_grid = self.param_grids[model_name]

        tscv = TimeSeriesSplit(n_splits=3)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=10,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        try:
            search.fit(X_train, y_train)
            print(f"Best CV F1 for {model_name}: {search.best_score_:.4f}")
            logger.info(f"Best CV F1 for {model_name}: {search.best_score_:.4f}")
            return search.best_estimator_, search.best_params_, search.best_score_
        except Exception as e:
            print(f"ERROR: Hyperparameter tuning failed for {model_name}: {e}")
            logger.error(f"Hyperparameter tuning failed for {model_name}: {e}")
            return base_model, {}, 0.0

class ModelEvaluator:
    """Handles model evaluation and results compilation"""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    @staticmethod
    def calculate_trading_metrics(y_pred, y_true, returns=None):
        """Calculate trading simulation metrics"""
        if returns is not None and len(returns) == len(y_pred):
            strategy_returns = np.where(y_pred == 1, returns, 0)
        else:
            # Fallback: simulate returns based on predictions vs actual
            strategy_returns = np.where(
                (y_pred == 1) & (y_true == 1), 0.01,
                np.where((y_pred == 1) & (y_true == 0), -0.01, 0)
            )

        total_return = np.sum(strategy_returns)
        volatility = np.std(strategy_returns) if len(strategy_returns) > 1 else 0
        sharpe_ratio = (np.mean(strategy_returns) / volatility * np.sqrt(252)) if volatility > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility
        }

    @staticmethod
    def plot_training_results(results_df, output_dir='/content/sample_data/Output'):
        """Create comprehensive training result plots"""
        os.makedirs(output_dir, exist_ok=True)

        # Set default font size
        plt.rcParams['font.size'] = 18

        # 1. Train vs Test Accuracy Comparison
        fig, axes = plt.subplots(2, 2, figsize=(22, 16))

        # Add main title
        fig.suptitle('Training Results Comparison', fontsize=24, y=0.98,fontweight='bold')

        # Accuracy comparison
        models = results_df['Model'].unique()
        datasets = results_df['Dataset'].unique()

        x = np.arange(len(models))
        width = 0.35

        for i, dataset in enumerate(datasets):
            data = results_df[results_df['Dataset'] == dataset]
            train_acc = data['Train_Accuracy'].values
            test_acc = data['Test_Accuracy'].values

            axes[0, i].bar(x - width/2, train_acc, width, label='Train Accuracy', alpha=0.8)
            axes[0, i].bar(x + width/2, test_acc, width, label='Test Accuracy', alpha=0.8)
            axes[0, i].set_xlabel('Models', fontsize=22)
            axes[0, i].set_ylabel('Accuracy', fontsize=22)
            axes[0, i].set_title(f'Train vs Test Accuracy - {dataset}', fontsize=22, pad=10)
            axes[0, i].set_xticks(x)
            axes[0, i].set_xticklabels(models, rotation=45, fontsize=22)
            axes[0, i].tick_params(axis='y', labelsize=22)
            axes[0, i].legend(fontsize=20)
            axes[0, i].grid(True, alpha=0.3)

            # Add value labels on bars
            for j, (train_val, test_val) in enumerate(zip(train_acc, test_acc)):
                axes[0, i].text(j - width/2, train_val + 0.01, f'{train_val:.3f}',
                              ha='center', va='bottom', fontsize=22)
                axes[0, i].text(j + width/2, test_val + 0.01, f'{test_val:.3f}',
                              ha='center', va='bottom', fontsize=22)

        # F1 Score comparison
        for i, dataset in enumerate(datasets):
            data = results_df[results_df['Dataset'] == dataset]
            train_f1 = data['Train_F1'].values
            test_f1 = data['Test_F1'].values

            axes[1, i].bar(x - width/2, train_f1, width, label='Train F1', alpha=0.8)
            axes[1, i].bar(x + width/2, test_f1, width, label='Test F1', alpha=0.8)
            axes[1, i].set_xlabel('Models', fontsize=22)
            axes[1, i].set_ylabel('F1 Score', fontsize=22)
            axes[1, i].set_title(f'Train vs Test F1 Score - {dataset}', fontsize=22, pad=10)
            axes[1, i].set_xticks(x)
            axes[1, i].set_xticklabels(models, rotation=45, fontsize=22)
            axes[1, i].tick_params(axis='y', labelsize=22)
            axes[1, i].legend(fontsize=20)
            axes[1, i].grid(True, alpha=0.3)

            # Add value labels on bars
            for j, (train_val, test_val) in enumerate(zip(train_f1, test_f1)):
                axes[1, i].text(j - width/2, train_val + 0.01, f'{train_val:.3f}',
                              ha='center', va='bottom', fontsize=22)
                axes[1, i].text(j + width/2, test_val + 0.01, f'{test_val:.3f}',
                              ha='center', va='bottom', fontsize=22)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        plt.savefig(f'{output_dir}/train_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
  
class StockPredictionPipeline:
    """Main pipeline for stock prediction analysis"""

    def __init__(self, df, selected_metadata_path, causal_metadata_path):
        self.df = df
        self.selected_metadata_path = selected_metadata_path
        self.causal_metadata_path = causal_metadata_path

        self.data_processor = DataProcessor(df)
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

        self.results = []
        self.best_params = {}

        self.y_true_dict = {}
        self.y_pred_dict = {}

    def load_feature_sets(self):
        """Load feature sets from metadata files"""
        print("Loading feature sets from metadata files")
        logger.info("Loading feature sets from metadata files")

        # Load metadata
        selected_metadata = MetadataLoader.load_metadata(self.selected_metadata_path)
        causal_metadata = MetadataLoader.load_metadata(self.causal_metadata_path)

        # Extract features
        selected_features = MetadataLoader.extract_features(selected_metadata)
        causal_features = MetadataLoader.extract_features(causal_metadata)

        return selected_features, causal_features

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting stock prediction analysis")
        logger.info("Starting stock prediction analysis")

        try:
            # Load and prepare data
            self.data_processor.prepare_data()

            # Load feature sets
            selected_features, causal_features = self.load_feature_sets()

            # Create feature sets
            feature_sets = self.data_processor.create_feature_sets(selected_features, causal_features)

            # Create datasets
            datasets = self.data_processor.create_datasets(feature_sets)

            if not datasets:
                print("ERROR: No valid datasets created")
                logger.error("No valid datasets created")
                return None

            # Train and evaluate models
            self._train_and_evaluate(datasets)

            # Create results DataFrame
            results_df = pd.DataFrame(self.results)

            # Generate plots
            self.evaluator.plot_training_results(results_df)

            #self._plot_regression_metrics(results_df)

            self._plot_confusion_matrices(
                                self.y_true_dict, self.y_pred_dict,
                                ['Random Forest', 'XGBoost', 'LightGBM'],
                                list(datasets.keys()),'/content/sample_data/Output')

            # Generate report
            self._generate_report(results_df)

            print("Analysis completed successfully")
            logger.info("Analysis completed successfully")
            return results_df

        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")
            logger.error(f"Analysis failed: {e}")
            return None

    def _train_and_evaluate(self, datasets):
        """Train and evaluate models on all datasets"""
        print("Training and evaluating models")
        logger.info("Training and evaluating models")

        model_names = {
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM'
        }

        for dataset_name, dataset in datasets.items():
            print(f"Processing {dataset_name} ({dataset['feature_count']} features)")
            logger.info(f"Processing {dataset_name} ({dataset['feature_count']} features)")

            X, y = dataset['X'], dataset['y']

            if len(X) < 50:
                print(f"WARNING: Insufficient data for {dataset_name}: {len(X)} samples")
                logger.warning(f"Insufficient data for {dataset_name}: {len(X)} samples")
                continue

            # Time series split
            test_size = max(int(len(X) * 0.2), 10)
            split_idx = len(X) - test_size

            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Get returns for trading simulation if available
            returns = None
            if 'Daily_Return' in self.df.columns:
                returns = self.df['Daily_Return'].iloc[split_idx:split_idx+len(y_test)].values

            for model_key, model_name in model_names.items():
                try:
                    # Tune hyperparameters
                    tuned_model, best_params, cv_score = self.model_trainer.tune_hyperparameters(
                        X_train, y_train, model_key
                    )

                    # Store best parameters
                    self.best_params[f"{dataset_name}_{model_key}"] = best_params

                    # Train and predict
                    tuned_model.fit(X_train, y_train)
                    y_train_pred = tuned_model.predict(X_train)
                    y_test_pred = tuned_model.predict(X_test)

                    # Calculate metrics
                    train_metrics = self.evaluator.calculate_metrics(y_train, y_train_pred)
                    test_metrics = self.evaluator.calculate_metrics(y_test, y_test_pred)
                    trading_metrics = self.evaluator.calculate_trading_metrics(
                        y_test_pred, y_test, returns
                    )

                    # Store predictions for confusion matrix
                    key = f"{dataset_name}_{model_key}"
                    self.y_true_dict[key] = y_test.values
                    self.y_pred_dict[key] = y_test_pred

                    # Calculate regression metrics if applicable
                    mae = mean_absolute_error(y_test, y_test_pred)
                    mse = mean_squared_error(y_test, y_test_pred)
                    r2 = r2_score(y_test, y_test_pred)

                    # Store results
                    result = {
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Features': dataset['feature_count'],
                        'Best_Params' : self.best_params,
                        'Train_Accuracy': train_metrics['accuracy'],
                        'Train_F1': train_metrics['f1'],
                        'Test_Accuracy': test_metrics['accuracy'],
                        'Test_Precision': test_metrics['precision'],
                        'Test_Recall': test_metrics['recall'],
                        'Test_F1': test_metrics['f1'],
                        'CV_F1_Score': cv_score,
                        'MAE': mae,
                        'MSE': mse,
                        'R2': r2,
                        'Total_Return': trading_metrics['total_return'],
                        'Sharpe_Ratio': trading_metrics['sharpe_ratio'],
                        'Overfitting_Score': train_metrics['accuracy'] - test_metrics['accuracy'],
                        'Train_Size': len(X_train),
                        'Test_Size': len(X_test),
                        'Description': dataset['description']
                    }

                    self.results.append(result)

                    print(f"{model_name}: CV F1={cv_score:.4f}, Test F1={test_metrics['f1']:.4f}, Best_Params={self.best_params}")
                    logger.info(f"{model_name}: CV F1={cv_score:.4f}, Test F1={test_metrics['f1']:.4f}")

                except Exception as e:
                    print(f"ERROR: Error training {model_name} on {dataset_name}: {e}")
                    logger.error(f"Error training {model_name} on {dataset_name}: {e}")

   

    def _plot_confusion_matrices(self, y_true_dict, y_pred_dict, model_names, dataset_names, output_dir='/content/sample_data/Output'):
      """Plot confusion matrices for all model-dataset combinations"""
      import matplotlib.pyplot as plt
      import seaborn as sns
      from sklearn.metrics import confusion_matrix

      os.makedirs(output_dir, exist_ok=True)

      # Set default font size
      plt.rcParams['font.size'] = 20

      n_models = len(model_names)
      n_datasets = len(dataset_names)

      # Create figure with column-wise layout (datasets as columns, models as rows)
      fig, axes = plt.subplots(n_models, n_datasets, figsize=(10*n_datasets, 6*n_models))

      # Add main title
      fig.suptitle('Confusion Matrix', fontsize=24, y=0.98,fontweight='bold')

      # Handle single row/column cases
      if n_models == 1 and n_datasets == 1:
          axes = [[axes]]
      elif n_models == 1:
          axes = [axes]
      elif n_datasets == 1:
          axes = [[ax] for ax in axes]

      # Plot confusion matrices (row = model, column = dataset)
      for i, model in enumerate(model_names):
          for j, dataset in enumerate(dataset_names):
              key = f"{dataset}_{model.lower().replace(' ', '_')}"

              if key in y_true_dict and key in y_pred_dict:
                  y_true = y_true_dict[key]
                  y_pred = y_pred_dict[key]

                  cm = confusion_matrix(y_true, y_pred)

                  # Create heatmap with larger annotation font
                  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Down (0)', 'Up (1)'],
                            yticklabels=['Down (0)', 'Up (1)'],
                            ax=axes[i][j],
                            annot_kws={'fontsize': 22},  # Set annotation font size
                            cbar_kws={'label': 'Count'})

                  accuracy = (cm[0,0] + cm[1,1]) / cm.sum()

                  # Set subplot title
                  axes[i][j].set_title(f'{model}\n{dataset}\nAcc: {accuracy:.3f}',
                                      fontsize=22, pad=10)
                  axes[i][j].set_xlabel('Predicted', fontsize=22)
                  axes[i][j].set_ylabel('Actual', fontsize=22)

                  # Set tick label font size
                  axes[i][j].tick_params(axis='both', labelsize=22)

                  # Adjust colorbar font size
                  cbar = axes[i][j].collections[0].colorbar
                  if cbar:
                      cbar.ax.tick_params(labelsize=22)
                      cbar.set_label('Count', fontsize=22)

      # Add column headers for clarity (optional)
      for j, dataset in enumerate(dataset_names):
          fig.text(0.5/n_datasets + j/n_datasets, 0.95, dataset.upper(),
                  ha='center', va='top', fontsize=22, weight='bold')

      plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for suptitle
      plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
      plt.close()

      print(f"Confusion matrices saved to {output_dir}/confusion_matrices.png")

    def _generate_report(self, results_df):
        """Generate comparison report"""
        print("Generating comparison report")
        logger.info("Generating comparison report")

        if results_df.empty:
            print("WARNING: No results to report")
            logger.warning("No results to report")
            return

        print("\nSTOCK PREDICTION ANALYSIS RESULTS")
        print("=" * 80)

        # Sort by Test F1 score
        results_sorted = results_df.sort_values('Test_F1', ascending=False)

        # Display results
        display_cols = ['Dataset', 'Model', 'Features', 'CV_F1_Score', 'Test_Accuracy',
                       'Test_F1', 'Sharpe_Ratio', 'MAE', 'MSE', 'R2','Overfitting_Score']
        print(results_sorted[display_cols].round(4).to_string(index=False))

        # Best model summary,
        if len(results_sorted) > 0:
            best_model = results_sorted.iloc[0]
            print(f"\nBEST PERFORMING MODEL:")
            print(f"Model: {best_model['Model']} on {best_model['Dataset']}")
            print(f"Test F1: {best_model['Test_F1']:.4f}")
            print(f"Test Accuracy: {best_model['Test_Accuracy']:.4f}")
            print(f"Sharpe Ratio: {best_model['Sharpe_Ratio']:.4f}")
        
def run_stock_prediction_analysis(df, selected_metadata_path, causal_metadata_path):
    """Main function to run stock prediction analysis"""
    pipeline = StockPredictionPipeline(df, selected_metadata_path, causal_metadata_path)
    results = pipeline.run_analysis()
    return pipeline, results

if __name__ == "__main__":
    print("Stock Market Prediction Analysis")
    print("Usage: pipeline, results = run_stock_prediction_analysis(df, 'selected_metadata.json', 'causal_metadata.json')")