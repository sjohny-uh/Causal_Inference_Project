import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import ta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class UKStockAnalyzer:
    """
    Analysis of UK stock market data incorporating
    macroeconomic factors and technical indicators
    """

    def __init__(self, tickers=None, start_date="2020-01-01"):
        """
        Initialize the analyzer with stock tickers and date range

        Parameters:
        -----------
        tickers : list, optional
            List of UK stock tickers (with .L suffix)
        start_date : str
            Start date for data collection
        """
        self.tickers = tickers or ['HSBA.L', 'BP.L', 'AZN.L', 'GSK.L', 'VOD.L']
        self.start_date = start_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.stock_df = None
        self.macro_df = None
        self.final_df = None

    def fetch_stock_data(self):
        """
        Download and process stock price data for all tickers

        Returns:
        --------
        pd.DataFrame: Processed stock data with consistent format
        """
        print(f"Fetching stock data for {len(self.tickers)} tickers...")

        try:
            # Download all data efficiently
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )

            # Process each ticker's data
            stock_dfs = []
            for ticker in self.tickers:
                try:
                    ticker_df = data[ticker].copy()
                    # Remove NaN rows
                    ticker_df = ticker_df.dropna()  

                    if not ticker_df.empty:
                        ticker_df['Date'] = ticker_df.index
                        ticker_df['Ticker'] = ticker
                        stock_dfs.append(ticker_df.reset_index(drop=True))
                        print(f"Successfully processed {ticker}: {len(ticker_df)} records")
                    else:
                        print(f" No data available for {ticker}")

                except KeyError:
                    print(f" Data not found for: {ticker}")
                    continue

            if stock_dfs:
                self.stock_df = pd.concat(stock_dfs, ignore_index=True)
                self.stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)
                print(f"\nStock data summary:")
                print(f"- Total records: {len(self.stock_df):,}")
                print(f"- Date range: {self.stock_df['Date'].min()} to {self.stock_df['Date'].max()}")
                print(f"- Tickers: {', '.join(self.stock_df['Ticker'].unique())}")
                return self.stock_df
            else:
                raise ValueError("No stock data could be retrieved")

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            raise

    def fetch_macroeconomic_data(self):
        """
        Collect and process all macroeconomic indicators

        Returns:
        --------
        pd.DataFrame: Combined macroeconomic dataset
        """
        print("\nFetching macroeconomic data...")

        # 1. GBP/USD Exchange Rate
        print("- Downloading GBP/USD exchange rate...")
        try:
            fx_data = yf.download("GBPUSD=X",
                                start="2015-01-01",
                                end=self.end_date,
                                progress=False)
            fx_data.columns = fx_data.columns.get_level_values(0)
            fx_data.reset_index(inplace=True)
            fx_data = fx_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            fx_data.columns.name = None
            fx_data.set_index('Date', inplace=True)
            fx_data = fx_data[['Close']].rename(columns={'Close': 'GBP/USD'})
            fx_data.reset_index(inplace=True)
            fx_data['Date'] = pd.to_datetime(fx_data['Date'])
            print(f"  FX data: {len(fx_data)} records")
        except Exception as e:
            print(f"  Error fetching FX data: {e}")
            fx_data = pd.DataFrame(columns=['Date', 'GBP/USD'])


        # 2. Consumer Price Index (CPI/Inflation)
        print("- Processing CPI data...")
        try:
            cpi_raw = pd.read_csv('cpih01-time-series-v57.csv')
            # Convert 'Time' to datetime (monthly frequency)
            cpi_raw['Date'] = pd.to_datetime(cpi_raw['Time'], format='%b-%y', errors='coerce')

            cpi_main = cpi_raw[cpi_raw['cpih1dim1aggid'] == 'CP00']

            # Rename and format
            cpi_main = cpi_main[['Time', 'v4_0']].copy()
            cpi_main.columns = ['Date', 'CPI']
            cpi_main['Date'] = pd.to_datetime(cpi_main['Date'], format='%b-%y')
            cpi_main['CPI'] = pd.to_numeric(cpi_main['CPI'], errors='coerce')

            # Resample to daily and forward fill
            cpi_daily = cpi_main.set_index('Date').resample('D').ffill().reset_index()

            print(f"CPI data: {len(cpi_daily)} records")
        except Exception as e:
            print(f"Error processing CPI data: {e}")
            cpi_daily = pd.DataFrame(columns=['Date', 'CPI'])

        # 3. Bank of England Base Rate
        print("- Processing Bank Rate data...")
        try:

            rate_df = pd.read_csv('/content/Bank_Rate_history_and_data_ Bank_of_England_Database.csv')
            # Rename columns
            rate_df.columns = ['Date Changed', 'Rate']
            rate_df['Date'] = pd.to_datetime(rate_df['Date Changed'], errors='coerce')
            rate_df.dropna(inplace=True)
            rate_df.drop(columns=['Date Changed'], inplace=True)

            # Convert to daily and fill missing values
            rate_daily = rate_df.set_index('Date').resample('D').ffill().reset_index()

            print(f"  ✓ Bank Rate data: {len(rate_daily)} records")
        except Exception as e:
            print(f"  ✗ Error processing Bank Rate data: {e}")
            rate_daily = pd.DataFrame(columns=['Date', 'Rate'])

        # 4. Unemployment Rate
        print("- Processing Unemployment data...")
        try:

            df = pd.read_csv("/content/Unemployment_series-180525.csv", skiprows=8)
            df.columns = ["Date", "UnemploymentRate"]

            # Keep only monthly data (rows where the date contains a month abbreviation, e.g., 'FEB')
            df_monthly = df[df['Date'].str.len() == 8]  # e.g., '1971 FEB' is 8 chars

            # Convert to datetime
            df_monthly['Date'] = pd.to_datetime(df_monthly['Date'], format="%Y %b")

            # Optional: sort and reset index
            df_monthly = df_monthly.sort_values('Date').reset_index(drop=True)
            df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])

            unemployment_daily = df_monthly.set_index('Date').resample('D').ffill().reset_index()
            unemployment_daily.sort_values('Date', inplace=True)

            print(f" Unemployment data: {len(unemployment_daily)} records")
        except Exception as e:
            print(f" Error processing Unemployment data: {e}")
            unemployment_daily = pd.DataFrame(columns=['Date', 'UnemploymentRate'])

        # Combine all macro data
        '''
        macro_datasets = [
            ('FX', fx_data),
            ('CPI', cpi_daily),
            ('Rate', rate_daily),
            ('Unemployment', unemployment_daily)
        ]
        '''
        print(fx_data.head())
        print(cpi_daily.head())
        print(rate_daily.head())
        print(unemployment_daily.head())

        for df in [cpi_daily, rate_daily, unemployment_daily, fx_data]:
            df['Date'] = pd.to_datetime(df['Date'])

        # Sort all dataframes
        cpi_daily = cpi_daily.sort_values('Date')
        rate_daily = rate_daily.sort_values('Date')
        unemployment_daily = unemployment_daily.sort_values('Date')
        fx_data = fx_data.sort_values('Date')

        self.macro_df=pd.DataFrame()

        # Merge CPI and Interest Rate
        self.macro_df = pd.merge_asof(unemployment_daily,cpi_daily, on='Date')

        # Merge Unemployment
        self.macro_df = pd.merge_asof(self.macro_df, rate_daily, on='Date')

        # Merge FX
        self.macro_df = pd.merge_asof( self.macro_df.sort_values('Date'), fx_data.sort_values('Date'))
        
        if self.macro_df is not None:
            print(f"\nMacroeconomic data summary:")
            print(f"- Total records: {len(self.macro_df):,}")
            print(f"- Date range: {self.macro_df['Date'].min()} to {self.macro_df['Date'].max()}")
            print(f"- Variables: {', '.join([col for col in self.macro_df.columns if col != 'Date'])}")

        return self.macro_df

    def merge_datasets(self):
        """
        Merge stock data with macroeconomic indicators using time-series joins

        Returns:
        --------
        pd.DataFrame: Combined dataset 
        """
        print("\nMerging datasets...")

        if self.stock_df is None or self.macro_df is None:
            raise ValueError("Stock and macro data must be fetched first")

        # Ensure proper datetime formats and sorting
        self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'])
        self.macro_df['Date'] = pd.to_datetime(self.macro_df['Date'])

        self.stock_df = self.stock_df.sort_values('Date')
        self.macro_df = self.macro_df.sort_values('Date')
        
        self.final_df = pd.merge_asof(self.stock_df,self.macro_df, on='Date')


        print(f" Merged dataset: {len(self.final_df):,} records")
        print(f" Features: {len(self.final_df.columns)} columns")

        return self.final_df

    def calculate_technical_indicators(self):
        """
        Calculate technical indicators for each stock

        Returns:
        --------
        pd.DataFrame: Dataset with technical indicators
        """
        print("\nCalculating technical indicators...")

        if self.final_df is None:
            raise ValueError("Data must be merged first")

        def apply_technical_analysis(group):
            """Apply technical indicators to a stock group"""
            group = group.copy().sort_values('Date')
            
            # Minimum data points for reliable indicators
            if len(group) < 50:  
                return group

            try:
                # Price and volume data
                close = group['Close']
                high = group['High']
                low = group['Low']
                volume = group['Volume']

                # MOMENTUM INDICATORS 
                # RSI (Relative Strength Index)
                group['RSI_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

                # MACD (Moving Average Convergence Divergence)
                macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
                group['MACD'] = macd.macd()
                group['MACD_Signal'] = macd.macd_signal()
                group['MACD_Histogram'] = macd.macd_diff()

                # Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
                group['Stoch_K'] = stoch.stoch()
                group['Stoch_D'] = stoch.stoch_signal()

                # TREND INDICATORS
                # Simple Moving Averages
                group['SMA_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
                group['SMA_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
                group['SMA_200'] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()

                # Exponential Moving Averages
                group['EMA_12'] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
                group['EMA_26'] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

                # Moving Average Signals
                group['SMA_20_50_Signal'] = np.where(group['SMA_20'] > group['SMA_50'], 1, 0)

                # VOLATILITY INDICATORS
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
                group['BB_Upper'] = bb.bollinger_hband()
                group['BB_Lower'] = bb.bollinger_lband()
                group['BB_Width'] = bb.bollinger_wband()
                group['BB_Position'] = (close - group['BB_Lower']) / (group['BB_Upper'] - group['BB_Lower'])

                # Average True Range (Volatility)
                group['ATR_14'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

                # VOLUME INDICATORS 
                # Volume Moving Average
                group['Volume_SMA_20'] = ta.trend.SMAIndicator(close=volume, window=20).sma_indicator()
                group['Volume_Ratio'] = volume / group['Volume_SMA_20']

                # PRICE ACTION INDICATORS
                # Daily Returns
                group['Daily_Return'] = close.pct_change()
                group['Daily_Return_Abs'] = group['Daily_Return'].abs()

                # Price momentum (multiple periods)
                group['Return_5D'] = close.pct_change(periods=5)
                group['Return_10D'] = close.pct_change(periods=10)
                group['Return_20D'] = close.pct_change(periods=20)

                # Rolling volatility
                group['Volatility_20D'] = group['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

                # Support/Resistance levels
                group['High_20D'] = high.rolling(window=20).max()
                group['Low_20D'] = low.rolling(window=20).min()
                group['Price_Position'] = (close - group['Low_20D']) / (group['High_20D'] - group['Low_20D'])

                # TARGET VARIABLES
                # Next day direction (main target)
                group['Target_Direction'] = group['Daily_Return'].shift(-1).apply(lambda x: 1 if x > 0 else 0)

                # Multi-period targets
                group['Target_5D_Return'] = group['Return_5D'].shift(-5)
                group['Target_10D_Return'] = group['Return_10D'].shift(-10)

                # Volatility-adjusted targets
                group['Target_Excess_Return'] = (group['Daily_Return'].shift(-1) - group['Daily_Return'].rolling(20).mean()) / group['Daily_Return'].rolling(20).std()

                return group

            except Exception as e:
                print(f"Error calculating indicators for group: {e}")
                return group

        # Apply technical analysis to each stock
        print("- Computing indicators by ticker...")
        self.final_df = self.final_df.groupby('Ticker').apply(apply_technical_analysis).reset_index(drop=True)

        # Remove rows with insufficient data
        initial_rows = len(self.final_df)
        self.final_df.dropna(subset=['RSI_14', 'MACD', 'SMA_50'], inplace=True)
        final_rows = len(self.final_df)

        print(f" Technical indicators calculated")
        print(f" Removed {initial_rows - final_rows:,} rows with insufficient data")
        print(f" Final dataset: {final_rows:,} records with {len(self.final_df.columns)} features")
        print(f" Final dataset: features: {self.final_df.columns}")

        return self.final_df


    # Add lag features
    def add_lag_features(self):
        """
        Add lag features for causal analysis with proper data preservation
        """
        print("Adding lag features for causal analysis...")

        if self.final_df is None:
            raise ValueError("Technical indicators must be calculated first")

        def apply_lags(group):
            group = group.copy().sort_values('Date')

            if len(group) < 30:
                return group

            try:
                # Macroeconomic lags
                for feature in ['UnemploymentRate', 'CPI', 'Rate', 'GBP/USD']:
                    if feature in group.columns:
                        for lag in [1, 5, 10, 20]:
                            col_name = f'{feature}_lag_{lag}' if feature != 'GBP/USD' else f'GBP_USD_lag_{lag}'
                            group[col_name] = group[feature].shift(lag)

                        change_name = f'{feature}_change' if feature != 'GBP/USD' else 'GBP_USD_change'
                        group[change_name] = group[feature].diff()

                # Technical lags
                for feature in ['RSI_14', 'MACD', 'BB_Position', 'ATR_14']:
                    if feature in group.columns:
                        for lag in [1, 3, 5]:
                            group[f'{feature}_lag_{lag}'] = group[feature].shift(lag)

                # Price action lags
                for feature in ['Daily_Return', 'Volatility_20D', 'Volume_Ratio']:
                    if feature in group.columns:
                        lags = [1, 2, 3, 5] if feature == 'Daily_Return' else [1, 3, 5]
                        for lag in lags:
                            group[f'{feature}_lag_{lag}'] = group[feature].shift(lag)

                return group

            except Exception as e:
                print(f"Error adding lags: {e}")
                return group

        # Apply lag features
        print("- Computing lag features by ticker...")
        self.final_df = self.final_df.groupby('Ticker').apply(apply_lags).reset_index(drop=True)

        # Conservative data cleaning
        lag_columns = [col for col in self.final_df.columns if 'lag_1' in col]

        if lag_columns:
            before = len(self.final_df)
            # Only remove rows where ALL short-term lags are missing
            self.final_df = self.final_df.dropna(subset=lag_columns, how='all')
            after = len(self.final_df)

            print(f" Lag features added")
            print(f" Removed {before - after:,} rows with insufficient lag data")
            print(f" Final dataset: {after:,} records with {len(self.final_df.columns)} features")

        return self.final_df

    def create_complete_analysis(self):
        """
        Generate complete visualizations and analysis
        """
        print("\nGenerating complete analysis...")

        if self.final_df is None:
            raise ValueError("Technical indicators must be calculated first")

        # Set up the plotting environment
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. Market Overview: Stock Prices Over Time
        self._plot_stock_prices()

        # 2. Macroeconomic Environment
        self._plot_macro_indicators()

        # 3. Technical Analysis Overview
        self._plot_technical_indicators()

        # 4. Correlation Analysis
        self._plot_correlation_analysis()

        # 5. Target Distribution Analysis
        self._plot_target_analysis()

        # 6. Feature Importance Analysis
        self._analyze_feature_importance()

    def _plot_stock_prices(self):
        """Plot individual stock price trends"""
        print("- Creating stock price visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        tickers = self.final_df['Ticker'].unique()

        for i, ticker in enumerate(tickers):
            if i >= len(axes):
                break

            ticker_data = self.final_df[self.final_df['Ticker'] == ticker].copy()

            ax = axes[i]
            ax.plot(ticker_data['Date'], ticker_data['Close'],
                   linewidth=2, alpha=0.8, label='Close Price')

            # Add moving averages if available
            if 'SMA_50' in ticker_data.columns:
                ax.plot(ticker_data['Date'], ticker_data['SMA_50'],
                       '--', alpha=0.7, label='SMA 50')

            ax.set_title(f'{ticker} Stock Price Trend', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (£)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(len(tickers), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.suptitle('UK Stock Price Trends (FTSE 100 Sample)',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.show()

    def _plot_macro_indicators(self):
        """Plot macroeconomic indicators"""
        print("- Creating macroeconomic visualizations...")

        # Get unique dates and macro data
        macro_subset = self.final_df[['Date', 'GBP/USD', 'CPI', 'Rate', 'UnemploymentRate']].drop_duplicates('Date').sort_values('Date')
        print(macro_subset.head())
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # GBP/USD Exchange Rate
        axes[0,0].plot(macro_subset['Date'], macro_subset['GBP/USD'],
                      color='darkblue', linewidth=2)
        axes[0,0].set_title('GBP/USD Exchange Rate', fontweight='bold')
        axes[0,0].set_ylabel('Exchange Rate')
        axes[0,0].grid(True, alpha=0.3)

        # Consumer Price Index
        axes[0,1].plot(macro_subset['Date'], macro_subset['CPI'],
                      color='darkred', linewidth=2)
        axes[0,1].set_title('Consumer Price Index (Inflation)', fontweight='bold')
        axes[0,1].set_ylabel('CPI')
        axes[0,1].grid(True, alpha=0.3)

        # Bank of England Base Rate
        axes[1,0].plot(macro_subset['Date'], macro_subset['Rate'],
                      color='darkgreen', linewidth=2)
        axes[1,0].set_title('Bank of England Base Rate', fontweight='bold')
        axes[1,0].set_ylabel('Interest Rate (%)')
        axes[1,0].grid(True, alpha=0.3)

        # Unemployment Rate
        axes[1,1].plot(macro_subset['Date'], macro_subset['UnemploymentRate'],
                      color='darkorange', linewidth=2)
        axes[1,1].set_title('UK Unemployment Rate', fontweight='bold')
        axes[1,1].set_ylabel('Unemployment Rate (%)')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('UK Macroeconomic Indicators',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.show()

    def _plot_technical_indicators(self):
        """Plot technical indicators for sample stock"""
        print("- Creating technical indicator visualizations...")

        # Use first ticker as example
        sample_ticker = self.final_df['Ticker'].unique()[0]
        sample_data = self.final_df[self.final_df['Ticker'] == sample_ticker].copy()
        sample_data = sample_data.tail(252)  # Last year of data

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Price and Moving Averages
        axes[0].plot(sample_data['Date'], sample_data['Close'],
                    label='Close Price', linewidth=2)
        axes[0].plot(sample_data['Date'], sample_data['SMA_20'],
                    label='SMA 20', alpha=0.7)
        axes[0].plot(sample_data['Date'], sample_data['SMA_50'],
                    label='SMA 50', alpha=0.7)
        axes[0].fill_between(sample_data['Date'],
                           sample_data['BB_Lower'], sample_data['BB_Upper'],
                           alpha=0.2, label='Bollinger Bands')
        axes[0].set_title(f'{sample_ticker} - Price and Moving Averages', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # RSI
        axes[1].plot(sample_data['Date'], sample_data['RSI_14'],
                    color='purple', linewidth=2)
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1].set_title('RSI (14-day)', fontweight='bold')
        axes[1].set_ylabel('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # MACD
        axes[2].plot(sample_data['Date'], sample_data['MACD'],
                    label='MACD', linewidth=2)
        axes[2].plot(sample_data['Date'], sample_data['MACD_Signal'],
                    label='Signal', alpha=0.7)
        axes[2].bar(sample_data['Date'], sample_data['MACD_Histogram'],
                   alpha=0.3, label='Histogram')
        axes[2].set_title('MACD', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_correlation_analysis(self):
        """Create complete correlation analysis"""
        print("- Creating correlation analysis...")

        # Select key features for correlation
        correlation_features = [
            'Close', 'Daily_Return', 'Volume',
            'RSI_14', 'MACD', 'BB_Width', 'ATR_14',
            'GBP/USD', 'CPI', 'Rate', 'UnemploymentRate',
            'Volatility_20D', 'Volume_Ratio'
        ]

        # Filter available columns
        available_features = [col for col in correlation_features if col in self.final_df.columns]

        correlation_data = self.final_df[available_features].corr()

        # Create correlation heatmap
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(correlation_data))
        sns.heatmap(correlation_data,
                   annot=True,
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   mask=mask,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})

        plt.title('Feature Correlation Matrix\n(Identifying Key Relationships)',
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def _plot_target_analysis(self):
        """Analyze target variable distributions"""
        print("- Creating target analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Target direction distribution
        target_counts = self.final_df['Target_Direction'].value_counts()
        axes[0,0].pie(target_counts.values,
                     labels=['Down (0)', 'Up (1)'],
                     autopct='%1.1f%%',
                     colors=['lightcoral', 'lightgreen'])
        axes[0,0].set_title('Daily Direction Distribution', fontweight='bold')

        # Daily returns histogram
        axes[0,1].hist(self.final_df['Daily_Return'].dropna(),
                      bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0,1].set_title('Daily Returns Distribution', fontweight='bold')
        axes[0,1].set_xlabel('Daily Return')
        axes[0,1].set_ylabel('Frequency')

        # Returns by ticker
        sns.boxplot(data=self.final_df, x='Ticker', y='Daily_Return', ax=axes[1,0])
        axes[1,0].set_title('Returns Distribution by Stock', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Volatility over time
        monthly_vol = self.final_df.groupby(self.final_df['Date'].dt.to_period('M'))['Daily_Return'].std()
        axes[1,1].plot(monthly_vol.index.astype(str), monthly_vol.values,
                      marker='o', linewidth=2)
        axes[1,1].set_title('Market Volatility Over Time', fontweight='bold')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_ylabel('Volatility')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def _analyze_feature_importance(self):
        """Analyze which features are most predictive"""
        print("- Analyzing feature importance...")

        # Calculate correlation with target variable
        numeric_cols = self.final_df.select_dtypes(include=[np.number]).columns
        target_correlations = []

        for col in numeric_cols:
            if col not in ['Target_Direction', 'Target_5D_Return', 'Target_10D_Return']:
                corr = self.final_df[col].corr(self.final_df['Target_Direction'])
                if not np.isnan(corr):
                    target_correlations.append((col, abs(corr)))

        # Sort by absolute correlation
        target_correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = target_correlations[:15]  # Top 15 features

        # Create feature importance plot
        features, importances = zip(*top_features)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances, color='steelblue', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Absolute Correlation with Next-Day Direction')
        plt.title('Feature Importance for Predicting Stock Direction\n(Based on Correlation Analysis)',
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        plt.show()

        # Print top insights
        print("\nKey Insights - Top Predictive Features:")
        for i, (feature, importance) in enumerate(top_features[:10], 1):
            print(f"{i:2d}. {feature:<20} | Correlation: {importance:.4f}")

    def generate_summary_report(self):
        """Generate complete summary report"""
        print("\n" + "="*80)
        print("UK STOCK MARKET ANALYSIS - SUMMARY REPORT")
        print("="*80)

        if self.final_df is None:
            print("Error: Analysis not completed. Run the full pipeline first.")
            return

        # Dataset Summary
        print(f"\n DATASET SUMMARY:")
        print(f"   • Total Records: {len(self.final_df):,}")
        print(f"   • Unique Stocks: {self.final_df['Ticker'].nunique()}")
        print(f"   • Date Range: {self.final_df['Date'].min().strftime('%Y-%m-%d')} to {self.final_df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   • Total Features: {len(self.final_df.columns)}")

        # Market Performance
        print(f"\n MARKET PERFORMANCE:")
        avg_returns = self.final_df.groupby('Ticker')['Daily_Return'].mean()
        volatilities = self.final_df.groupby('Ticker')['Daily_Return'].std()

        for ticker in avg_returns.index:
            print(f"   • {ticker:<8} | Avg Daily Return: {avg_returns[ticker]:.4f} | Volatility: {volatilities[ticker]:.4f}")

        # Direction Prediction Accuracy
        direction_accuracy = self.final_df['Target_Direction'].value_counts(normalize=True)
        print(f"\n MARKET DIRECTION:")
        print(f"   • Up Days: {direction_accuracy.get(1, 0):.1%}")
        print(f"   • Down Days: {direction_accuracy.get(0, 0):.1%}")

        # Macro Environment Summary
        print(f"\n MACROECONOMIC ENVIRONMENT:")
        if 'GBP/USD' in self.final_df.columns:
            gbp_avg = self.final_df['GBP/USD'].mean()
            gbp_std = self.final_df['GBP/USD'].std()
            print(f"   • GBP/USD Average: {gbp_avg:.4f} (±{gbp_std:.4f})")

        if 'Rate' in self.final_df.columns:
            rate_avg = self.final_df['Rate'].mean()
            print(f"   • Average BoE Rate: {rate_avg:.2f}%")

        if 'UnemploymentRate' in self.final_df.columns:
            unemp_avg = self.final_df['UnemploymentRate'].mean()
            print(f"   • Average Unemployment: {unemp_avg:.2f}%")

        print("\n" + "="*80)

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

        return analyzer

    except Exception as e:
        print(f"\n ERROR: {e}")
        print("Please ensure you have the required data files or adjust the data sources.")
        return None
