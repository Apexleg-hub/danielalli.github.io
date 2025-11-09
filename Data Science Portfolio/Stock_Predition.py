import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, symbol, period="2y"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        
    def download_data(self):
        """Download stock data using yfinance"""
        print(f"Downloading data for {self.symbol}...")
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for symbol {self.symbol}")
            
        print(f"Downloaded {len(self.data)} days of data")
        return self.data
    
    def calculate_rsi(self, window=14):
        """Calculate Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, span=20):
        """Calculate Exponential Moving Average"""
        return self.data['Close'].ewm(span=span, adjust=False).mean()
    
    def calculate_macd(self):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, window=20):
        """Calculate Bollinger Bands"""
        sma = self.data['Close'].rolling(window=window).mean()
        std = self.data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
    def calculate_stochastic_oscillator(self, window=14):
        """Calculate Stochastic Oscillator"""
        low_min = self.data['Low'].rolling(window=window).min()
        high_max = self.data['High'].rolling(window=window).max()
        stoch = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        return stoch
    
    def engineer_features(self):
        """Create technical indicators as features"""
        print("Engineering features...")
        
        # Price-based features
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['High_Low_Ratio'] = self.data['High'] / self.data['Low']
        self.data['Open_Close_Ratio'] = self.data['Open'] / self.data['Close']
        
        # Volume features
        self.data['Volume_Change'] = self.data['Volume'].pct_change()
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=5).mean()
        
        # RSI
        self.data['RSI'] = self.calculate_rsi()
        self.data['RSI_SMA'] = self.data['RSI'].rolling(window=5).mean()
        
        # EMA
        self.data['EMA_20'] = self.calculate_ema(20)
        self.data['EMA_50'] = self.calculate_ema(50)
        self.data['Price_vs_EMA20'] = self.data['Close'] / self.data['EMA_20']
        
        # MACD
        macd, signal, histogram = self.calculate_macd()
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = signal
        self.data['MACD_Histogram'] = histogram
        
        # Bollinger Bands
        upper_bb, lower_bb = self.calculate_bollinger_bands()
        self.data['BB_Upper'] = upper_bb
        self.data['BB_Lower'] = lower_bb
        self.data['BB_Position'] = (self.data['Close'] - lower_bb) / (upper_bb - lower_bb)
        
        # Stochastic Oscillator
        self.data['Stochastic'] = self.calculate_stochastic_oscillator()
        
        # Moving averages
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        
        # Create target variable (1 if price goes up next day, 0 otherwise)
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        
        # Drop NaN values
        self.data = self.data.dropna()
        
        # Define feature columns
        feature_columns = [
            'Price_Change', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Volume_Change', 'Volume_SMA', 'RSI', 'RSI_SMA',
            'Price_vs_EMA20', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'Stochastic', 'SMA_5', 'SMA_10', 'SMA_20'
        ]
        
        self.features = self.data[feature_columns]
        self.target = self.data['Target']
        
        print(f"Created {len(feature_columns)} features for {len(self.data)} samples")
        return self.features, self.target
    
    def prepare_data(self, test_size=0.2):
        """Split and scale the data"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, shuffle=False, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate multiple models"""
        print("\nTraining models...")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'SVM':
                # SVM requires scaled data and might need parameter tuning
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
        
        return results
    
    def plot_results(self, results, y_test):
        """Plot model performance and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall']
        
        metric_values = {metric: [results[model][metric] for model in models] for metric in metrics}
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(x + i * width, metric_values[metric], width, label=metric)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Scores')
        axes[0, 0].set_title('Model Comparison')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confusion matrices
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1] if i == 0 else axes[1, i-1])
            if i == 0:
                axes[0, 1].set_title(f'{model_name} - Confusion Matrix')
            else:
                axes[1, i-1].set_title(f'{model_name} - Confusion Matrix')
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.features.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title('Random Forest - Feature Importance')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        try:
            # Download and prepare data
            self.download_data()
            self.engineer_features()
            
            # Prepare training and test data
            X_train, X_test, y_train, y_test = self.prepare_data()
            
            # Train models
            results = self.train_models(X_train, X_test, y_train, y_test)
            
            # Plot results
            self.plot_results(results, y_test)
            
            # Print detailed classification reports
            print("\n" + "="*50)
            print("DETAILED CLASSIFICATION REPORTS")
            print("="*50)
            
            for name, result in results.items():
                print(f"\n{name}:")
                print(classification_report(y_test, result['predictions']))
            
            return results
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Analyze Apple stock
    predictor = StockPredictor("AAPL", period="2y")
    results = predictor.run_analysis()
    
    # You can also try other stocks:
    # predictor = StockPredictor("GOOGL", period="2y")
    # predictor = StockPredictor("MSFT", period="2y")
    # predictor = StockPredictor("TSLA", period="2y")