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

        # 1. Train vs Test Accuracy Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

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
            axes[0, i].set_xlabel('Models')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].set_title(f'Train vs Test Accuracy - {dataset}')
            axes[0, i].set_xticks(x)
            axes[0, i].set_xticklabels(models, rotation=45)
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

        # F1 Score comparison
        for i, dataset in enumerate(datasets):
            data = results_df[results_df['Dataset'] == dataset]
            train_f1 = data['Train_F1'].values
            test_f1 = data['Test_F1'].values

            axes[1, i].bar(x - width/2, train_f1, width, label='Train F1', alpha=0.8)
            axes[1, i].bar(x + width/2, test_f1, width, label='Test F1', alpha=0.8)
            axes[1, i].set_xlabel('Models')
            axes[1, i].set_ylabel('F1 Score')
            axes[1, i].set_title(f'Train vs Test F1 Score - {dataset}')
            axes[1, i].set_xticks(x)
            axes[1, i].set_xticklabels(models, rotation=45)
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/train_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Performance Metrics Heatmap
        '''
        fig, ax = plt.subplots(figsize=(12, 8))

        pivot_data = results_df.pivot_table(
            values=['Test_Accuracy', 'Test_F1', 'Sharpe_Ratio'],
            index='Model',
            columns='Dataset'
        )

        import seaborn as sns
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('Model Performance Heatmap')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        '''
        # Melt the DataFrame to long format
        melted = results_df.melt(id_vars=['Model', 'Dataset'], 
                                 value_vars=['Test_Accuracy', 'Test_F1', 'Sharpe_Ratio'],
                                 var_name='Metric', value_name='Value')

        # Create a new column combining Metric and Dataset for unique columns
        melted['Metric_Dataset'] = melted['Metric'] + '_' + melted['Dataset']

        # Pivot to get a flat table
        pivot_data = melted.pivot(index='Model', columns='Metric_Dataset', values='Value')

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('Model Performance Heatmap')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
        
        # 3. Sharpe Ratio vs Accuracy Scatter
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['red', 'blue', 'green']
        for i, model in enumerate(models):
            model_data = results_df[results_df['Model'] == model]
            ax.scatter(model_data['Test_Accuracy'], model_data['Sharpe_Ratio'],
                      c=colors[i], label=model, s=100, alpha=0.7)

        ax.set_xlabel('Test Accuracy')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Risk-Adjusted Performance vs Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sharpe_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training result plots saved to {output_dir}")


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

            self._plot_regression_metrics(results_df)

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

                    print(f"{model_name}: CV F1={cv_score:.4f}, Test F1={test_metrics['f1']:.4f}")
                    logger.info(f"{model_name}: CV F1={cv_score:.4f}, Test F1={test_metrics['f1']:.4f}")

                except Exception as e:
                    print(f"ERROR: Error training {model_name} on {dataset_name}: {e}")
                    logger.error(f"Error training {model_name} on {dataset_name}: {e}")


    def _plot_regression_metrics(self,results_df, output_dir='/content/sample_data/Output'):
        """Create comprehensive regression metrics visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. MSE Comparison by Model and Dataset
        pivot_mse = results_df.pivot(index='Model', columns='Dataset', values='MSE')
        sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='Reds', ax=axes[0,0])
        axes[0,0].set_title('Mean Squared Error (MSE) Heatmap\nLower is Better')

        # 2. MAE Comparison by Model and Dataset
        pivot_mae = results_df.pivot(index='Model', columns='Dataset', values='MAE')
        sns.heatmap(pivot_mae, annot=True, fmt='.4f', cmap='Oranges', ax=axes[0,1])
        axes[0,1].set_title('Mean Absolute Error (MAE) Heatmap\nLower is Better')

        # 3. R² Comparison by Model and Dataset
        pivot_r2 = results_df.pivot(index='Model', columns='Dataset', values='R2')
        sns.heatmap(pivot_r2, annot=True, fmt='.4f', cmap='Blues', ax=axes[0,2])
        axes[0,2].set_title('R² Score Heatmap\nHigher is Better')

        # 4. MSE vs MAE Scatter Plot
        colors = {'Random Forest': 'red', 'XGBoost': 'blue', 'LightGBM': 'green'}
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            axes[1,0].scatter(model_data['MSE'], model_data['MAE'],
                             c=colors.get(model, 'black'), label=model, s=100, alpha=0.7)

        axes[1,0].set_xlabel('Mean Squared Error (MSE)')
        axes[1,0].set_ylabel('Mean Absolute Error (MAE)')
        axes[1,0].set_title('MSE vs MAE Relationship')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 5. R² vs Test Accuracy
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            axes[1,1].scatter(model_data['R2'], model_data['Test_Accuracy'],
                             c=colors.get(model, 'black'), label=model, s=100, alpha=0.7)

        axes[1,1].set_xlabel('R² Score')
        axes[1,1].set_ylabel('Test Accuracy')
        axes[1,1].set_title('R² vs Classification Accuracy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # 6. Combined Metrics Bar Chart
        metrics_summary = results_df.groupby('Model')[['MSE', 'MAE', 'R2']].mean()

        x = range(len(metrics_summary.index))
        width = 0.25

        # Normalize metrics for comparison (0-1 scale)
        mse_norm = 1 - (metrics_summary['MSE'] / metrics_summary['MSE'].max())  # Invert MSE
        mae_norm = 1 - (metrics_summary['MAE'] / metrics_summary['MAE'].max())  # Invert MAE
        r2_norm = metrics_summary['R2'] / metrics_summary['R2'].max() if metrics_summary['R2'].max() > 0 else metrics_summary['R2']

        axes[1,2].bar([i - width for i in x], mse_norm, width, label='MSE (inverted)', alpha=0.8)
        axes[1,2].bar(x, mae_norm, width, label='MAE (inverted)', alpha=0.8)
        axes[1,2].bar([i + width for i in x], r2_norm, width, label='R²', alpha=0.8)

        axes[1,2].set_xlabel('Models')
        axes[1,2].set_ylabel('Normalized Score (Higher = Better)')
        axes[1,2].set_title('Normalized Regression Metrics Comparison')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(metrics_summary.index, rotation=45)
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression_metrics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Additional plot: Metrics by Feature Count
        fig, ax = plt.subplots(figsize=(12, 8))

        for dataset in results_df['Dataset'].unique():
            dataset_data = results_df[results_df['Dataset'] == dataset]
            ax.scatter(dataset_data['Features'], dataset_data['R2'],
                      label=f'{dataset} - R²', s=100, alpha=0.7)
            ax.scatter(dataset_data['Features'], 1-dataset_data['MSE'],
                      label=f'{dataset} - MSE (inverted)', s=100, alpha=0.7, marker='^')

        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Performance Score')
        ax.set_title('Regression Performance vs Feature Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Regression metrics plots saved to {output_dir}")


    def _plot_confusion_matrices(self,y_true_dict, y_pred_dict, model_names, dataset_names, output_dir='/content/sample_data/Output'):
        """Plot confusion matrices for all model-dataset combinations"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        os.makedirs(output_dir, exist_ok=True)

        n_cols = len(model_names)
        n_rows = len(dataset_names)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for i, dataset in enumerate(dataset_names):
            for j, model in enumerate(model_names):
                key = f"{dataset}_{model.lower().replace(' ', '_')}"

                if key in y_true_dict and key in y_pred_dict:
                    y_true = y_true_dict[key]
                    y_pred = y_pred_dict[key]

                    cm = confusion_matrix(y_true, y_pred)

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=['Down (0)', 'Up (1)'],
                               yticklabels=['Down (0)', 'Up (1)'],
                               ax=axes[i][j])

                    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
                    axes[i][j].set_title(f'{model}\n{dataset}\nAcc: {accuracy:.3f}')
                    axes[i][j].set_xlabel('Predicted')
                    axes[i][j].set_ylabel('Actual')

        plt.tight_layout()
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
        '''
        # Feature set comparison
        print(f"\nFEATURE SET COMPARISON:")
        comparison = results_df.groupby('Dataset').agg({
            'Test_F1': 'mean',
            'Test_Accuracy': 'mean',
            'Sharpe_Ratio': 'mean',
            'Features': 'first'
        }).round(4)
        print(comparison.to_string())
        '''
def run_stock_prediction_analysis(df, selected_metadata_path, causal_metadata_path):
    """Main function to run stock prediction analysis"""
    pipeline = StockPredictionPipeline(df, selected_metadata_path, causal_metadata_path)
    results = pipeline.run_analysis()
    return pipeline, results

if __name__ == "__main__":
    print("Stock Market Prediction Analysis")
    print("Usage: pipeline, results = run_stock_prediction_analysis(df, 'selected_metadata.json', 'causal_metadata.json')")
