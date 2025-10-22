"""
AI-Driven Stock Price Prediction Module
Uses technical indicators to train ML models for price direction prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AIPricePredictor:
    """
    AI model for predicting stock price direction using technical indicators.
    Ensures no future data leakage during training and prediction.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor with specified model type.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from technical indicators already calculated.
        Uses only past data - NO FUTURE LEAKAGE.
        
        Args:
            df: DataFrame with technical indicators already calculated
            
        Returns:
            DataFrame with prepared features
        """
        df = df.copy()
        
        # Target: Next candle direction (1=up, 0=down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Price-based features (using only past data)
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_3'] = df['close'].pct_change(3)
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ratio'] = df['volume'] / df['vol_sma_14']
        
        # Technical indicator features (already calculated by your script)
        feature_list = [
            # Price changes
            'price_change', 'price_change_2', 'price_change_3',
            
            # Volume
            'volume_change', 'volume_ratio',
            
            # MACD
            'MACD', 'MACD_Signal',
            
            # RSI
            'RSI',
            
            # Ichimoku
            'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
            
            # ATR
            'atr_values',
            
            # ADX
            'adx', 'plus_di', 'minus_di',
            
            # Aroon
            'aroon_up', 'aroon_down',
            
            # EMA
            'ema_6', 'ema_12', 'ema_18',
            
            # Heikin Ashi
            'ha_close', 'ha_open',
            
            # VWAP
            'vwap',
            
            # Hawkes
            'v_hawk'
        ]
        
        # Add relative position features
        df['price_above_ema6'] = (df['close'] > df['ema_6']).astype(int)
        df['price_above_ema12'] = (df['close'] > df['ema_12']).astype(int)
        df['price_above_vwap'] = (df['close'] > df['vwap']).astype(int)
        df['tenkan_above_kijun'] = (df['tenkan_sen'] > df['kijun_sen']).astype(int)
        df['macd_above_signal'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        feature_list.extend([
            'price_above_ema6', 'price_above_ema12', 'price_above_vwap',
            'tenkan_above_kijun', 'macd_above_signal'
        ])
        
        # Store feature columns
        self.feature_columns = [col for col in feature_list if col in df.columns]
        
        # Drop rows with NaN values
        df = df.dropna(subset=self.feature_columns + ['target'])
        
        return df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the model using time-series split (no future data leakage).
        
        Args:
            df: DataFrame with features prepared
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        print("🤖 Preparing features for AI model training...")
        df_prepared = self.prepare_features(df)
        
        # Features and target
        X = df_prepared[self.feature_columns]
        y = df_prepared['target']
        
        # Time-series split (maintains temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train model
        print(f"🚀 Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"✅ Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'y_test': y_test,
            'y_pred': y_test_pred,
            'test_dates': df_prepared.index[split_idx:]
        }
    
    def predict_next(self, df: pd.DataFrame, current_idx: int) -> Tuple[int, float]:
        """
        Predict next price direction for a specific timestamp.
        Uses ONLY data up to current_idx (no future leakage).
        
        Args:
            df: DataFrame with features
            current_idx: Current position in dataframe
            
        Returns:
            Tuple of (prediction, probability)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction!")
        
        # Get features for current timestamp only
        current_features = df.iloc[current_idx][self.feature_columns].values.reshape(1, -1)
        
        # Scale and predict
        current_scaled = self.scaler.transform(current_features)
        prediction = self.model.predict(current_scaled)[0]
        probability = self.model.predict_proba(current_scaled)[0][prediction]
        
        return int(prediction), float(probability)
    
    def backtest_predictions(self, df: pd.DataFrame, start_idx: int = None) -> pd.DataFrame:
        """
        Run backtest predictions on historical data.
        
        Args:
            df: DataFrame with features
            start_idx: Starting index for backtest (default: 80% of data)
            
        Returns:
            DataFrame with predictions and actual values
        """
        if not self.trained:
            raise ValueError("Model must be trained before backtesting!")
        
        df_prepared = self.prepare_features(df)
        
        if start_idx is None:
            start_idx = int(len(df_prepared) * 0.8)
        
        results = []
        
        print(f"🔄 Running backtest from index {start_idx} to {len(df_prepared)}...")
        
        for idx in range(start_idx, len(df_prepared) - 1):
            # Get actual values
            current_price = df_prepared.iloc[idx]['close']
            next_price = df_prepared.iloc[idx + 1]['close']
            actual_direction = 1 if next_price > current_price else 0
            
            # Predict
            pred_direction, confidence = self.predict_next(df_prepared, idx)
            
            results.append({
                'timestamp': df_prepared.index[idx],
                'current_price': current_price,
                'next_price': next_price,
                'actual_direction': actual_direction,
                'predicted_direction': pred_direction,
                'confidence': confidence,
                'correct': actual_direction == pred_direction
            })
        
        results_df = pd.DataFrame(results)
        accuracy = results_df['correct'].mean()
        
        print(f"✅ Backtest Accuracy: {accuracy*100:.2f}%")
        
        return results_df
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """
        Plot model performance metrics and predictions.
        
        Args:
            results: Results dictionary from train()
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AI Price Prediction Model Performance', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Feature Importance (Top 15)
        top_features = results['feature_importance'].head(15)
        axes[0, 1].barh(top_features['feature'], top_features['importance'])
        axes[0, 1].set_title('Top 15 Feature Importances')
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].invert_yaxis()
        
        # 3. Accuracy Comparison
        accuracies = [results['train_accuracy'], results['test_accuracy']]
        axes[1, 0].bar(['Training', 'Testing'], accuracies, color=['#2ecc71', '#3498db'])
        axes[1, 0].set_title('Model Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[1, 0].text(i, v + 0.01, f'{v*100:.2f}%', ha='center', fontweight='bold')
        
        # 4. Prediction Distribution
        pred_counts = pd.Series(results['y_pred']).value_counts()
        axes[1, 1].pie(pred_counts.values, labels=['Down', 'Up'], autopct='%1.1f%%', 
                       colors=['#e74c3c', '#2ecc71'], startangle=90)
        axes[1, 1].set_title('Prediction Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.savefig('ai_model_performance.png', dpi=300, bbox_inches='tight')
            print("📊 Plot saved to ai_model_performance.png")
        
        plt.show()
    
    def plot_price_predictions(self, backtest_results: pd.DataFrame, save_path: str = None):
        """
        Plot predicted vs actual price movements.
        
        Args:
            backtest_results: DataFrame from backtest_predictions()
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Price Prediction vs Actual', fontsize=16, fontweight='bold')
        
        # 1. Price movement with predictions
        axes[0].plot(backtest_results['timestamp'], backtest_results['next_price'], 
                     label='Actual Price', linewidth=2, alpha=0.7)
        
        # Color code predictions
        correct_pred = backtest_results[backtest_results['correct']]
        wrong_pred = backtest_results[~backtest_results['correct']]
        
        axes[0].scatter(correct_pred['timestamp'], correct_pred['next_price'], 
                       color='green', marker='o', s=30, label='Correct Prediction', alpha=0.6)
        axes[0].scatter(wrong_pred['timestamp'], wrong_pred['next_price'], 
                       color='red', marker='x', s=30, label='Wrong Prediction', alpha=0.6)
        
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Price')
        axes[0].set_title('Actual Price with Prediction Accuracy Markers')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Cumulative accuracy over time
        backtest_results['cumulative_accuracy'] = backtest_results['correct'].expanding().mean()
        axes[1].plot(backtest_results['timestamp'], backtest_results['cumulative_accuracy'] * 100,
                    linewidth=2, color='#3498db')
        axes[1].axhline(y=50, color='red', linestyle='--', label='Random Guess (50%)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Cumulative Prediction Accuracy Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.savefig('price_predictions.png', dpi=300, bbox_inches='tight')
            print("📊 Plot saved to price_predictions.png")
        
        plt.show()
    
    def generate_report(self, results: Dict[str, Any], backtest_results: pd.DataFrame = None):
        """
        Generate comprehensive performance report.
        
        Args:
            results: Results from train()
            backtest_results: Results from backtest_predictions()
        """
        print("\n" + "="*70)
        print("📊 AI PRICE PREDICTION MODEL - PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n🤖 Model Type: {self.model_type.upper()}")
        print(f"📈 Number of Features: {len(self.feature_columns)}")
        
        print(f"\n✅ ACCURACY METRICS:")
        print(f"   Training Accuracy:   {results['train_accuracy']*100:.2f}%")
        print(f"   Testing Accuracy:    {results['test_accuracy']*100:.2f}%")
        
        if backtest_results is not None:
            backtest_accuracy = backtest_results['correct'].mean()
            print(f"   Backtest Accuracy:   {backtest_accuracy*100:.2f}%")
        
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(results['classification_report'])
        
        print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for idx, row in results['feature_importance'].head(10).iterrows():
            print(f"   {idx+1}. {row['feature']:20s} - {row['importance']:.4f}")
        
        if backtest_results is not None:
            up_pred = (backtest_results['predicted_direction'] == 1).sum()
            down_pred = (backtest_results['predicted_direction'] == 0).sum()
            print(f"\n📊 PREDICTION DISTRIBUTION:")
            print(f"   Up predictions:   {up_pred} ({up_pred/len(backtest_results)*100:.1f}%)")
            print(f"   Down predictions: {down_pred} ({down_pred/len(backtest_results)*100:.1f}%)")
            
            avg_confidence = backtest_results['confidence'].mean()
            print(f"\n💪 AVERAGE PREDICTION CONFIDENCE: {avg_confidence*100:.2f}%")
        
        print("\n" + "="*70)


def main():
    """
    Main function demonstrating the AI price predictor.
    """
    print("="*70)
    print("🚀 AI-DRIVEN PRICE PREDICTION SYSTEM")
    print("="*70)
    
    # This will be integrated with your existing quant.py
    # For now, this is a standalone demonstration
    
    print("\n⚠️  This module is ready to be integrated with your trading system!")
    print("📝 Usage:")
    print("   1. Your quant.py calculates all technical indicators")
    print("   2. Pass the dataframe to AIPricePredictor")
    print("   3. Train model and get predictions")
    print("\n✅ No changes needed to your indicator calculations!")


if __name__ == "__main__":
    main()
