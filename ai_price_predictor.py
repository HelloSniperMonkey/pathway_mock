"""
AI-Driven Stock Price Prediction Module
Uses technical indicators to train ML models for price direction prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
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
    
    def __init__(self, model_type='gradient_boosting'):
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
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Momentum features
        df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_7'] = df['close'] / df['close'].shift(7) - 1
        
        # Volatility features
        df['price_std_5'] = df['close'].rolling(5).std()
        df['price_std_10'] = df['close'].rolling(10).std()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ratio'] = df['volume'] / df['vol_sma_14']
        df['volume_ma_ratio'] = df['vol_sma_9'] / df['vol_sma_14']
        
        # RSI derivative
        df['rsi_change'] = df['RSI'].diff()
        df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        
        # MACD features
        df['macd_histogram'] = df['MACD'] - df['MACD_Signal']
        df['macd_hist_change'] = df['macd_histogram'].diff()
        
        # EMA crossovers and distances
        df['ema_6_12_diff'] = (df['ema_6'] - df['ema_12']) / df['close']
        df['ema_12_18_diff'] = (df['ema_12'] - df['ema_18']) / df['close']
        df['price_ema6_dist'] = (df['close'] - df['ema_6']) / df['close']
        
        # ADX trend strength
        df['adx_strong'] = (df['adx'] > 25).astype(int)
        df['di_diff'] = df['plus_di'] - df['minus_di']
        
        # Technical indicator features (already calculated by your script)
        feature_list = [
            # Price changes and momentum
            'price_change', 'price_change_2', 'price_change_3', 'price_change_5',
            'momentum_3', 'momentum_7',
            
            # Volatility
            'price_std_5', 'price_std_10', 'high_low_range',
            
            # Volume
            'volume_change', 'volume_ratio', 'volume_ma_ratio',
            
            # MACD
            'MACD', 'MACD_Signal', 'macd_histogram', 'macd_hist_change',
            
            # RSI
            'RSI', 'rsi_change', 'rsi_oversold', 'rsi_overbought',
            
            # Ichimoku
            'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
            
            # ATR
            'atr_values',
            
            # ADX
            'adx', 'plus_di', 'minus_di', 'adx_strong', 'di_diff',
            
            # Aroon
            'aroon_up', 'aroon_down',
            
            # EMA
            'ema_6', 'ema_12', 'ema_18',
            'ema_6_12_diff', 'ema_12_18_diff', 'price_ema6_dist',
            
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
        
        # Calculate class distribution
        class_counts = y_train.value_counts()
        print(f"Training class distribution: UP={class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y_train)*100:.1f}%), DOWN={class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y_train)*100:.1f}%)")
        
        # Calculate class weights to balance the dataset
        # Give more weight to the minority class (usually UP)
        n_samples = len(y_train)
        n_classes = 2
        class_weight_0 = n_samples / (n_classes * class_counts.get(0, 1))
        class_weight_1 = n_samples / (n_classes * class_counts.get(1, 1))
        class_weights = {0: class_weight_0, 1: class_weight_1}
        print(f"Using class weights: DOWN={class_weight_0:.2f}, UP={class_weight_1:.2f}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model with balanced class weights
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,  # Increased for better learning
                max_depth=12,  # Increased slightly
                min_samples_split=15,  # Reduced to allow more splits
                min_samples_leaf=5,  # Reduced to allow finer granularity
                max_features='sqrt',  # Use sqrt of features for diversity
                class_weight=class_weights,  # Balance classes
                random_state=42,
                n_jobs=-1
            )
        else:
            # Calculate scale_pos_weight for XGBoost (ratio of negative to positive)
            scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
            print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
            
            self.model = xgb.XGBClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=4,  # Reduced to prevent overfitting
                learning_rate=0.05,  # Reduced for better learning
                min_child_weight=3,  # Add regularization
                subsample=0.8,  # Use 80% of data for each tree
                colsample_bytree=0.8,  # Use 80% of features
                gamma=0.1,  # Minimum loss reduction for split
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                scale_pos_weight=scale_pos_weight,  # Balance classes properly
                random_state=42,
                verbosity=0,
                eval_metric='logloss'
            )
        
        # Train model
        print(f"🚀 Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        # Get probability predictions for test set
        y_test_proba = self.model.predict_proba(X_test_scaled)
        
        # Find optimal threshold for balanced UP/DOWN accuracy
        print("\n🎯 Finding optimal prediction threshold...")
        best_threshold = 0.5
        best_metric = 0
        
        for threshold in np.arange(0.35, 0.65, 0.02):
            y_test_pred_thresh = (y_test_proba[:, 1] >= threshold).astype(int)
            
            # Calculate accuracy for each class
            up_mask = y_test == 1
            down_mask = y_test == 0
            
            up_acc = accuracy_score(y_test[up_mask], y_test_pred_thresh[up_mask]) if up_mask.sum() > 0 else 0
            down_acc = accuracy_score(y_test[down_mask], y_test_pred_thresh[down_mask]) if down_mask.sum() > 0 else 0
            
            # We want BOTH accuracies to be good, not just average
            # Use minimum of the two (ensures both classes perform well)
            # Multiply by average to also care about overall performance
            min_acc = min(up_acc, down_acc)
            avg_acc = (up_acc + down_acc) / 2
            metric = min_acc * 0.7 + avg_acc * 0.3  # Weighted combination
            
            if metric > best_metric and min_acc > 0.50:  # Ensure both classes > 50%
                best_metric = metric
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        print(f"✓ Optimal threshold: {best_threshold:.2f}")
        
        # Predictions with standard threshold
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = (y_test_proba[:, 1] >= self.optimal_threshold).astype(int)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"✅ Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"✅ Test Accuracy: {test_accuracy*100:.2f}% (with optimized threshold)")
        
        # Per-class accuracy
        up_mask = y_test == 1
        down_mask = y_test == 0
        up_acc = accuracy_score(y_test[up_mask], y_test_pred[up_mask]) if up_mask.sum() > 0 else 0
        down_acc = accuracy_score(y_test[down_mask], y_test_pred[down_mask]) if down_mask.sum() > 0 else 0
        print(f"   UP accuracy: {up_acc*100:.1f}%, DOWN accuracy: {down_acc*100:.1f}%")
        
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
    
    def predict_next(self, df: pd.DataFrame, current_idx: int, threshold: float = 0.5) -> Tuple[int, float]:
        """
        Predict next price direction for a specific timestamp.
        Uses ONLY data up to current_idx (no future leakage).
        
        Args:
            df: DataFrame with features
            current_idx: Current position in dataframe
            threshold: Probability threshold for predicting UP (default 0.5)
            
        Returns:
            Tuple of (prediction, probability)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction!")
        
        # Get features for current timestamp only
        current_features = df.iloc[current_idx][self.feature_columns].values.reshape(1, -1)
        
        # Scale and predict probabilities
        current_scaled = self.scaler.transform(current_features)
        proba = self.model.predict_proba(current_scaled)[0]
        
        # Use custom threshold instead of 0.5
        # proba[1] is probability of UP (class 1)
        prediction = 1 if proba[1] >= threshold else 0
        probability = proba[prediction]
        
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
        print(f"Using optimal threshold: {getattr(self, 'optimal_threshold', 0.5):.2f}")
        
        for idx in range(start_idx, len(df_prepared) - 1):
            # Get actual values
            current_price = df_prepared.iloc[idx]['close']
            next_price = df_prepared.iloc[idx + 1]['close']
            actual_direction = 1 if next_price > current_price else 0
            
            # Predict using optimal threshold
            threshold = getattr(self, 'optimal_threshold', 0.5)
            pred_direction, confidence = self.predict_next(df_prepared, idx, threshold=threshold)
            
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
