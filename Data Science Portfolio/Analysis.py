import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def download_data(self):
        """Download stock data using yfinance"""
        print(f"Downloading data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.data)} rows of data")
        return self.data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def engineer_features(self):
        """Create technical indicators as features"""
        print("Engineering features...")
        df = self.data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        
        # Momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Target: 1 if price goes up next day, 0 otherwise
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        self.data = df
        print(f"Features engineered. Dataset size: {len(df)}")
        return df
    
    def prepare_data(self):
        """Prepare features and target for modeling"""
        feature_cols = ['Returns', 'High_Low_Pct', 'Close_Open_Pct', 
                       'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
                       'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
                       'BB_Width', 'Volume_Change', 'Volume_MA_5',
                       'Momentum_5', 'Momentum_10']
        
        X = self.data[feature_cols]
        y = self.data['Target']
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train SVM and Random Forest models"""
        print("\nTraining models...")
        
        # SVM
        print("Training SVM...")
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        self.models['SVM'] = svm_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                         random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        self.models['RandomForest'] = rf_model
        
        print("Models trained successfully!")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name} Model:")
            print("-" * 40)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Down', 'Up']))
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        
        return results
    
    def plot_results(self, results, y_test):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Stock Price Movement Prediction - {self.ticker}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        
        axes[0, 0].bar(models, accuracies, color=['#3498db', '#2ecc71'])
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Plot 2: Precision vs Recall
        precisions = [results[m]['precision'] for m in models]
        recalls = [results[m]['recall'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, precisions, width, label='Precision', color='#e74c3c')
        axes[0, 1].bar(x + width/2, recalls, width, label='Recall', color='#f39c12')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: Confusion Matrix for SVM
        cm_svm = confusion_matrix(y_test, results['SVM']['predictions'])
        im1 = axes[1, 0].imshow(cm_svm, cmap='Blues', aspect='auto')
        axes[1, 0].set_title('SVM Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(['Down', 'Up'])
        axes[1, 0].set_yticklabels(['Down', 'Up'])
        
        for i in range(2):
            for j in range(2):
                axes[1, 0].text(j, i, cm_svm[i, j], ha='center', 
                               va='center', color='white', fontweight='bold')
        
        # Plot 4: Confusion Matrix for Random Forest
        cm_rf = confusion_matrix(y_test, results['RandomForest']['predictions'])
        im2 = axes[1, 1].imshow(cm_rf, cmap='Greens', aspect='auto')
        axes[1, 1].set_title('Random Forest Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(['Down', 'Up'])
        axes[1, 1].set_yticklabels(['Down', 'Up'])
        
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(j, i, cm_rf[i, j], ha='center', 
                               va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self):
        """Execute the complete prediction pipeline"""
        # Download data
        self.download_data()
        
        # Engineer features
        self.engineer_features()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Plot results
        self.plot_results(results, y_test)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = StockPredictor(
        ticker='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    # Run the complete pipeline
    results = predictor.run_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)