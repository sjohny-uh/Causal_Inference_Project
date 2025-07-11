import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import json
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, df, target_variable='Target_Direction'):
        self.df = df.copy()
        self.target = target_variable
        self.selected_features = []
        self.feature_importance_scores = {}
        self.modeling_df = None

    def prepare_data(self):
        """Prepare features and target"""
        exclude_cols = ['Date', 'Ticker', 'Target_Direction', 'Target_5D_Return',
                       'Target_10D_Return', 'Target_Excess_Return']

        feature_cols = [col for col in self.df.columns
                       if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]

        X = self.df[feature_cols].copy()
        y = self.df[self.target].copy()

        # Handle missing values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Align X and y
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        return X, y, feature_cols

    def select_features(self, method='lasso', max_features=30):
        """Feature selection"""
        X, y, all_features = self.prepare_data()

        if method == 'lasso':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
            lasso.fit(X_scaled, y)

            # Get feature importance scores
            self.feature_importance_scores = dict(zip(X.columns, np.abs(lasso.coef_)))

            # Select non-zero features
            mask = lasso.coef_ != 0
            selected_features = X.columns[mask].tolist()

            # Fallback if no features selected
            if len(selected_features) == 0:
                sorted_features = sorted(self.feature_importance_scores.items(),
                                       key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:min(max_features, len(sorted_features))]]

            # Limit to max_features
            elif len(selected_features) > max_features:
                sorted_features = sorted(self.feature_importance_scores.items(),
                                       key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:max_features]]

        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            # Get feature importance scores
            self.feature_importance_scores = dict(zip(X.columns, rf.feature_importances_))

            # Select top features
            sorted_features = sorted(self.feature_importance_scores.items(),
                                   key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:max_features]]

        self.selected_features = selected_features
        return selected_features

    def categorize_features(self):
        """Categorize features by type"""
        macro_features = []
        technical_features = []
        price_features = []
        other_features = []

        for feature in self.selected_features:
            feature_lower = feature.lower()
            if any(x in feature_lower for x in ['unemployment', 'cpi', 'rate', 'gbp', 'usd']):
                macro_features.append(feature)
            elif any(x in feature_lower for x in ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema', 'stoch']):
                technical_features.append(feature)
            elif any(x in feature_lower for x in ['return', 'volatility', 'volume']):
                price_features.append(feature)
            else:
                other_features.append(feature)

        return macro_features, technical_features, price_features, other_features

    def create_metadata(self, method='lasso'):
        """Create metadata in required format"""
        # Get all features
        exclude_cols = ['Date', 'Ticker', 'Target_Direction', 'Target_5D_Return',
                       'Target_10D_Return', 'Target_Excess_Return']
        all_features = [col for col in self.df.columns
                       if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]

        # Get non-selected features
        non_selected_features = [f for f in all_features if f not in self.selected_features]

        # Categorize features
        macro_selected, technical_selected, price_selected, other_selected = self.categorize_features()

        # Feature priority ranking
        feature_priority_ranking = sorted(self.feature_importance_scores.keys(),
                                        key=lambda x: self.feature_importance_scores[x],
                                        reverse=True)

        # Prepare modeling dataset
        self.modeling_df = self.df.copy()

        metadata = {
            'selected_features': self.selected_features,
            'non_selected_features': non_selected_features,
            'macro_selected_features': macro_selected,
            'technical_selected_features': technical_selected,
            'price_selected_features': price_selected,
            'other_selected_features': other_selected,
            'feature_priority_ranking': feature_priority_ranking,
            'feature_importance_scores': self.feature_importance_scores,
            'target_variable': self.target,
            'modeling_approach': 'feature_selection',
            'selection_method': method,
            'analysis_summary': {
                'total_features_tested': len(all_features),
                'significant_relationships': len(self.selected_features),
                'dataset_shape': list(self.modeling_df.shape),
                'analysis_date': str(pd.Timestamp.now().isoformat()),
                'selection_pipeline': [method + '_selection']
            }
        }

        return metadata

    def save_metadata(self, filepath='feature_selection_metadata.json', method='lasso'):
        """Save metadata to JSON file"""
        metadata = self.create_metadata(method)

       # Store in modeling_df.attrs for compatibility
        self.modeling_df.attrs['selected_features'] = self.selected_features
        self.modeling_df.attrs['selected_features'] = metadata['selected_features']
        self.modeling_df.attrs['macro_selected_features'] = metadata['macro_selected_features']
        self.modeling_df.attrs['technical_selected_features'] = metadata['technical_selected_features']
        self.modeling_df.attrs['price_selected_features'] = metadata['price_selected_features']
        self.modeling_df.attrs['other_selected_features'] = metadata['other_selected_features']
        self.modeling_df.attrs['feature_priority_ranking'] = metadata['feature_priority_ranking']
        self.modeling_df.attrs['feature_importance_scores'] = metadata['feature_importance_scores']

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Metadata saved to {filepath}")
        print(f"Selected {len(self.selected_features)} features using {method}")

        return metadata

    def plot_feature_importance(self, metadata_path='feature_selection_metadata.json', top_n=10, save_path='/content/sample_data/Output/feature_importance_plot.png'):
      """
      Plots the top N feature importances from the feature_selection_metadata.json file.

      Args:
          metadata_path (str): Path to the feature_selection_metadata.json file.
          top_n (int): Number of top features to plot.
          save_path (str): If provided, saves the plot to this path.
      """
      # Load metadata
      with open(metadata_path, 'r') as f:
          metadata = json.load(f)

      # Extract and sort importances
      importances_dict = metadata['feature_importance_scores']
      sorted_items = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
      top_features = [item[0] for item in sorted_items]
      top_importances = [item[1] for item in sorted_items]

      # Plot
      plt.figure(figsize=(8, 5))
      bars = plt.barh(top_features[::-1], top_importances[::-1], color='skyblue')
      plt.xlabel('Importance Score')
      plt.title(f'Top {top_n} Features by Random Forest Importance')
      ax = plt.gca()  # Get current axes
      ax.tick_params(axis='y', labelsize=8)  # Set y-axis font size

      plt.tight_layout()
      if save_path:
          plt.savefig(save_path, dpi=300)
      #plt.show()

def run_feature_selection(df, target_variable='Target_Direction', method='lasso', max_features=30):
    """Run feature selection analysis"""
    selector = FeatureSelector(df, target_variable)
    selected_features = selector.select_features(method=method, max_features=max_features)
    metadata = selector.save_metadata(method=method)
    selector.plot_feature_importance('feature_selection_metadata.json', 30, '/content/sample_data/Output/feature_importance_plot')

    print(f"\nFeature Selection Results:")
    print(f"Method: {method}")
    print(f"Total features tested: {metadata['analysis_summary']['total_features_tested']}")
    print(f"Features selected: {len(selected_features)}")
    print(f"\nTop 10 selected features:")
    for i, feature in enumerate(metadata['feature_priority_ranking'][:10], 1):
        score = metadata['feature_importance_scores'][feature]
        print(f"{i:2d}. {feature:<30} | Score: {score:.4f}")

    return selector, metadata

if __name__ == "__main__":
    print("Feature Selection Script")
    print("Usage: selector, metadata = run_feature_selection(df, method='lasso')")
