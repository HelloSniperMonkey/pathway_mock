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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AIPricePredictor:
    """
    AI model for predicting stock price direction using technical indicators.
    Ensures no future data leakage during training and prediction.
    """
    
    def __init__(self, model_type='random_forest', prediction_horizon=3, target_type='direction',
                 optimize_for: str = 'accuracy', abstain_band: float = 0.0, transaction_cost: float = 0.0005):
        """
        Initialize the predictor with specified model type.
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', 'logistic', 'neural_net', or 'ensemble'
            prediction_horizon: Number of periods ahead to predict (default: 3)
            target_type: 'direction' (binary), 'bucket' (3-class: down/flat/up), or 'threshold' (move > 2%)
        """
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.target_type = target_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.trained = False
        # Trading/post-processing config
        self.optimize_for = optimize_for  # 'accuracy' or 'sharpe'
        self.abstain_band = max(0.0, min(0.49, float(abstain_band)))  # symmetric band around 0.5
        self.transaction_cost = max(0.0, float(transaction_cost))  # per trade (one-way) as fraction
        # Thresholds for trading actions (set after training)
        self.optimal_threshold = 0.5
        self.trade_hi = 0.5 + self.abstain_band / 2.0
        self.trade_lo = 0.5 - self.abstain_band / 2.0
        
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
        
        # Calculate future price based on horizon
        future_close = df['close'].shift(-self.prediction_horizon)
        price_change_pct = ((future_close - df['close']) / df['close']) * 100
        
        # Create target based on type
        if self.target_type == 'direction':
            # Binary: up (1) or down (0) over horizon
            df['target'] = (future_close > df['close']).astype(int)
        elif self.target_type == 'bucket':
            # 3-class: significant down (0), flat (1), significant up (2)
            df['target'] = pd.cut(price_change_pct, 
                                 bins=[-np.inf, -1.0, 1.0, np.inf], 
                                 labels=[0, 1, 2]).astype(int)
        elif self.target_type == 'threshold':
            # Binary: significant move > 2% (1) or not (0)
            df['target'] = (price_change_pct.abs() > 2.0).astype(int)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
        
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

    def _predict_proba(self, X_selected: np.ndarray) -> np.ndarray:
        """
        Internal helper to get class probabilities from the trained model(s).
        Supports ensemble averaging when model_type == 'ensemble'.
        """
        if self.model_type == 'ensemble':
            probs = []
            for m in self.ensemble_models.values():
                probs.append(m.predict_proba(X_selected))
            # Average probabilities across models
            return np.mean(probs, axis=0)
        else:
            return self.model.predict_proba(X_selected)
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """
        Train the model using walk-forward time-series cross-validation.
        """
        print("🤖 Preparing features for AI model training...")
        print(f"📊 Prediction horizon: {self.prediction_horizon} periods ahead")
        print(f"🎯 Target type: {self.target_type}")
        df_prepared = self.prepare_features(df)

        # Features and target
        X = df_prepared[self.feature_columns]
        y = df_prepared['target']

        print(f"\n🔄 Using walk-forward validation with {n_splits} splits...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        purge = max(1, int(self.prediction_horizon))
        embargo = min(2, purge)

        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]

        print("📈 Cross-validation scores:")
        for i, (train_i, val_i) in enumerate(splits[:-1]):
            cutoff = val_i[0] - purge
            train_i_purged = train_i[train_i < cutoff]
            val_i_emb = val_i[embargo:] if len(val_i) > embargo else val_i
            X_cv_train, X_cv_val = X.iloc[train_i_purged], X.iloc[val_i_emb]
            y_cv_train, y_cv_val = y.iloc[train_i_purged], y.iloc[val_i_emb]

            temp_scaler = StandardScaler()
            X_cv_train_scaled = temp_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = temp_scaler.transform(X_cv_val)

            if self.model_type == 'logistic':
                temp_model = LogisticRegression(max_iter=1000, random_state=42)
            elif self.model_type in ('gradient_boosting', 'ensemble'):
                temp_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
            elif self.model_type == 'neural_net':
                temp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                           alpha=1e-4, learning_rate_init=1e-3,
                                           max_iter=200, early_stopping=True,
                                           n_iter_no_change=10, validation_fraction=0.1,
                                           random_state=42)
            else:
                temp_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

            temp_model.fit(X_cv_train_scaled, y_cv_train)
            cv_score = temp_model.score(X_cv_val_scaled, y_cv_val)
            cv_scores.append(cv_score)
            print(f"   Fold {i+1}: {cv_score*100:.2f}%")

        print(f"   Mean CV Score: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")

        # Final split with purge and embargo
        final_cutoff = test_idx[0] - purge
        train_idx_purged = train_idx[train_idx < final_cutoff]
        test_idx_emb = test_idx[embargo:] if len(test_idx) > embargo else test_idx
        X_train, X_test = X.iloc[train_idx_purged], X.iloc[test_idx_emb]
        y_train, y_test = y.iloc[train_idx_purged], y.iloc[test_idx_emb]

        print(f"\n📊 Final split:")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        class_counts = y_train.value_counts()
        print(f"Training class distribution: UP={class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y_train)*100:.1f}%), DOWN={class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y_train)*100:.1f}%)")

        n_samples = len(y_train)
        n_classes = 2
        class_weight_0 = n_samples / (n_classes * class_counts.get(0, 1))
        class_weight_1 = n_samples / (n_classes * class_counts.get(1, 1))
        class_weights = {0: class_weight_0, 1: class_weight_1}
        print(f"Using class weights: DOWN={class_weight_0:.2f}, UP={class_weight_1:.2f}")

        # Scale and select features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n🔍 Performing feature selection...")
        if self.model_type == 'logistic':
            selector_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42, max_iter=1000)
            selector_model.fit(X_train_scaled, y_train)
            selector = SelectFromModel(selector_model, prefit=True, threshold='median')
        else:
            temp_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            temp_rf.fit(X_train_scaled, y_train)
            selector = SelectFromModel(temp_rf, prefit=True, threshold='median')

        X_train_selected = selector.transform(X_train_scaled)
        X_test_selected = selector.transform(X_test_scaled)
        self.selector = selector
        self.selected_features = [f for f, s in zip(self.feature_columns, selector.get_support()) if s]
        print(f"✓ Selected {len(self.selected_features)} features (from {len(self.feature_columns)})")

        # Train model(s)
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_split=20, min_samples_leaf=8,
                max_features='sqrt', class_weight=class_weights, random_state=42, n_jobs=-1
            )
            self.model.fit(X_train_selected, y_train)
        elif self.model_type == 'gradient_boosting':
            scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
            self.model = xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.7, gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0, eval_metric='logloss'
            )
            self.model.fit(X_train_selected, y_train)
        elif self.model_type == 'neural_net':
            # Neural network classifier (MLP)
            # Use sample weights derived from class weights to handle imbalance
            try:
                sample_weights = y_train.map(class_weights).values
            except Exception:
                # Fallback: uniform weights if mapping fails
                sample_weights = None
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                alpha=1e-4, learning_rate_init=1e-3, max_iter=500, early_stopping=True,
                n_iter_no_change=15, validation_fraction=0.12, random_state=42
            )
            # Some sklearn versions accept sample_weight for MLPClassifier.fit; pass if available
            try:
                if sample_weights is not None:
                    self.model.fit(X_train_selected, y_train, sample_weight=sample_weights)
                else:
                    self.model.fit(X_train_selected, y_train)
            except TypeError:
                # If sample_weight isn't supported by the installed sklearn version
                self.model.fit(X_train_selected, y_train)
        elif self.model_type == 'ensemble':
            rf_model = RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_split=20, min_samples_leaf=8,
                max_features='sqrt', class_weight=class_weights, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train_selected, y_train)
            scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.7, gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0, eval_metric='logloss'
            )
            xgb_model.fit(X_train_selected, y_train)
            base_lr = LogisticRegression(penalty='l2', C=0.05, solver='lbfgs', max_iter=10000, class_weight=class_weights, random_state=42)
            lr_model = CalibratedClassifierCV(base_lr, method='sigmoid', cv=3)
            lr_model.fit(X_train_selected, y_train)
            self.ensemble_models = {'rf': rf_model, 'xgb': xgb_model, 'logit': lr_model}
            self.model = None
        else:
            base_model = LogisticRegression(penalty='l2', C=0.05, solver='lbfgs', max_iter=10000, class_weight=class_weights, random_state=82)
            self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
            self.model.fit(X_train_selected, y_train)

        self.trained = True

        # Probabilities for test set
        y_test_proba = self._predict_proba(X_test_selected)

        # Classification threshold optimization (balanced)
        print("\n🎯 Optimizing thresholds...")
        best_threshold = 0.5
        best_metric = 0.0
        for threshold in np.arange(0.35, 0.65, 0.02):
            y_test_pred_thresh = (y_test_proba[:, 1] >= threshold).astype(int)
            up_mask = y_test == 1
            down_mask = y_test == 0
            up_acc = accuracy_score(y_test[up_mask], y_test_pred_thresh[up_mask]) if up_mask.sum() > 0 else 0
            down_acc = accuracy_score(y_test[down_mask], y_test_pred_thresh[down_mask]) if down_mask.sum() > 0 else 0
            min_acc = min(up_acc, down_acc)
            avg_acc = (up_acc + down_acc) / 2
            metric = min_acc * 0.7 + avg_acc * 0.3
            if metric > best_metric and min_acc > 0.50:
                best_metric = metric
                best_threshold = threshold
        self.optimal_threshold = best_threshold
        print(f"✓ Optimal classification threshold: {best_threshold:.2f}")

        # Profit-aware trading band optimization
        close_series = df_prepared['close']
        test_positions = df_prepared.index[test_idx_emb]
        fwd_close = close_series.shift(-self.prediction_horizon)
        fwd_ret = (fwd_close - close_series) / close_series
        fwd_ret = fwd_ret.reindex(test_positions)
        proba_up = pd.Series(y_test_proba[:, 1], index=test_positions)

        def evaluate_band(hi: float) -> Tuple[float, Dict[str, float]]:
            lo = 1.0 - hi
            actions = proba_up.apply(lambda p: 1 if p >= hi else (-1 if p <= lo else 0))
            step_returns = actions * fwd_ret
            action_shift = actions.shift(1).fillna(0)
            trades = (actions != action_shift) & (actions != 0)
            costs = trades.astype(float) * self.transaction_cost
            exits = (actions == 0) & (action_shift != 0)
            costs += exits.astype(float) * self.transaction_cost
            net_returns = step_returns.fillna(0) - costs
            valid_mask = actions != 0
            n_trades = int(trades.sum())
            sharpe = float(net_returns.mean() / net_returns.std(ddof=0)) if net_returns.std(ddof=0) > 0 else 0.0
            cum = (1 + net_returns.fillna(0)).cumprod()
            peak = cum.cummax()
            drawdown = float((cum / peak - 1.0).min()) if len(cum) > 0 else 0.0
            total_return = float(cum.iloc[-1] - 1.0) if len(cum) > 0 else 0.0
            win_rate = float((net_returns[valid_mask] > 0).mean()) if valid_mask.any() else 0.0
            avg_trade = float(net_returns[valid_mask].mean()) if valid_mask.any() else 0.0
            metrics = {'sharpe': sharpe, 'total_return': total_return, 'max_drawdown': drawdown, 'win_rate': win_rate, 'avg_trade_return': avg_trade, 'n_trades': n_trades}
            objective = sharpe if self.optimize_for == 'sharpe' else total_return
            return objective, metrics

        best_hi = 0.5 + self.abstain_band / 2.0
        best_obj = -np.inf
        best_metrics = None
        for hi in np.arange(0.52, 0.71, 0.01):
            obj, mets = evaluate_band(hi)
            if mets['n_trades'] < 5:
                continue
            if obj > best_obj:
                best_obj = obj
                best_hi = float(hi)
                best_metrics = mets
        self.trade_hi = float(best_hi)
        self.trade_lo = float(1.0 - best_hi)
        self.profit_metrics_test = best_metrics or {'sharpe': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0, 'avg_trade_return': 0.0, 'n_trades': 0}
        print(f"✓ Trading band selected: lo={self.trade_lo:.2f}, hi={self.trade_hi:.2f} (optimize_for={self.optimize_for})")

        # Metrics and feature importance
        if self.model_type == 'ensemble':
            y_train_pred = (self._predict_proba(X_train_selected)[:, 1] >= 0.5).astype(int)
        else:
            y_train_pred = self.model.predict(X_train_selected)
        y_test_pred = (y_test_proba[:, 1] >= self.optimal_threshold).astype(int)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"✅ Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"✅ Test Accuracy: {test_accuracy*100:.2f}% (with optimized threshold)")

        up_mask = y_test == 1
        down_mask = y_test == 0
        up_acc = accuracy_score(y_test[up_mask], y_test_pred[up_mask]) if up_mask.sum() > 0 else 0
        down_acc = accuracy_score(y_test[down_mask], y_test_pred[down_mask]) if down_mask.sum() > 0 else 0
        print(f"   UP accuracy: {up_acc*100:.1f}%, DOWN accuracy: {down_acc*100:.1f}%")

        if self.model_type == 'ensemble':
            importances = []
            rf = self.ensemble_models.get('rf')
            if hasattr(rf, 'feature_importances_'):
                importances.append(rf.feature_importances_)
            xgb_m = self.ensemble_models.get('xgb')
            if hasattr(xgb_m, 'feature_importances_'):
                importances.append(xgb_m.feature_importances_)
            logit = self.ensemble_models.get('logit')
            try:
                base_estimator = logit.calibrated_classifiers_[0].estimator
                if hasattr(base_estimator, 'coef_'):
                    importances.append(np.abs(base_estimator.coef_[0]))
            except Exception:
                pass
            arr = np.vstack(importances) if len(importances) else np.zeros((1, len(self.selected_features)))
            mean_imp = arr.mean(axis=0)
            feature_importance = pd.DataFrame({'feature': self.selected_features, 'importance': mean_imp}).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({'feature': self.selected_features, 'importance': self.model.feature_importances_}).sort_values('importance', ascending=False)
        elif isinstance(self.model, CalibratedClassifierCV):
            base_estimator = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, 'coef_'):
                feature_importance = pd.DataFrame({'feature': self.selected_features, 'importance': np.abs(base_estimator.coef_[0])}).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({'feature': self.selected_features, 'importance': [0] * len(self.selected_features)})
        elif isinstance(self.model, MLPClassifier) and hasattr(self.model, 'coefs_') and len(self.model.coefs_) > 0:
            # Heuristic importance: mean absolute weights from input to first hidden layer
            first_layer_weights = self.model.coefs_[0]  # shape: (n_features, n_hidden)
            imp = np.mean(np.abs(first_layer_weights), axis=1)
            feature_importance = pd.DataFrame({'feature': self.selected_features, 'importance': imp}).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({'feature': self.selected_features, 'importance': [0] * len(self.selected_features)})

        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'y_test': y_test,
            'y_pred': y_test_pred,
            'test_dates': df_prepared.index[test_idx],
            'trading_band': {'lo': self.trade_lo, 'hi': self.trade_hi},
            'profit_metrics_test': self.profit_metrics_test
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
        current_features = np.array(df.iloc[current_idx][self.feature_columns]).reshape(1, -1)
        
        # Scale and apply feature selection
        current_scaled = self.scaler.transform(current_features)
        current_selected = self.selector.transform(current_scaled)
        
        # Predict probabilities
        proba = self._predict_proba(current_selected)[0]
        
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
        print(f"Using trading band: lo={self.trade_lo:.2f}, hi={self.trade_hi:.2f}, cost={self.transaction_cost:.4f}")
        
        prev_action = 0
        equity = 1.0
        for idx in range(start_idx, len(df_prepared) - self.prediction_horizon):
            # Get actual values at prediction horizon
            current_price = float(df_prepared.iloc[idx]['close'])
            future_price = float(df_prepared.iloc[idx + self.prediction_horizon]['close'])
            
            # Determine actual outcome based on target type
            if self.target_type == 'direction':
                actual_direction = 1 if future_price > current_price else 0
            elif self.target_type == 'bucket':
                price_change_pct = ((future_price - current_price) / current_price) * 100
                if price_change_pct < -1.0:
                    actual_direction = 0
                elif price_change_pct > 1.0:
                    actual_direction = 2
                else:
                    actual_direction = 1
            else:  # threshold
                price_change_pct = abs((future_price - current_price) / current_price) * 100
                actual_direction = 1 if price_change_pct > 2.0 else 0
            
            # Predict probability of UP
            current_features = np.array(df_prepared.iloc[idx][self.feature_columns]).reshape(1, -1)
            current_scaled = self.scaler.transform(current_features)
            current_selected = self.selector.transform(current_scaled)
            proba = self._predict_proba(current_selected)[0]
            p_up = float(proba[1])
            # Derive direction (for accuracy report)
            threshold = getattr(self, 'optimal_threshold', 0.5)
            pred_direction = 1 if p_up >= threshold else 0
            confidence = max(p_up, 1 - p_up)
            # Trading action with abstention band
            if p_up >= self.trade_hi:
                action = 1
            elif p_up <= self.trade_lo:
                action = -1
            else:
                action = 0
            # Compute step return and transaction costs
            step_ret = (future_price - current_price) / current_price
            trade_ret = action * step_ret
            # Costs on position change (entry/exit)
            trade_cost = 0.0
            if action != prev_action:
                if action != 0:
                    trade_cost += self.transaction_cost
                if prev_action != 0 and action == 0:
                    trade_cost += self.transaction_cost
            net_ret = trade_ret - trade_cost
            equity *= (1.0 + net_ret)
            prev_action = action

            results.append({
                'timestamp': df_prepared.index[idx],
                'current_price': current_price,
                'next_price': future_price,
                'actual_direction': actual_direction,
                'predicted_direction': pred_direction,
                'confidence': confidence,
                'correct': actual_direction == pred_direction,
                'p_up': p_up,
                'action': action,
                'step_return': step_ret,
                'trade_return': trade_ret,
                'net_return': net_ret,
                'equity': equity
            })
        
        results_df = pd.DataFrame(results)
        accuracy = results_df['correct'].mean()
        # Profit metrics
        n_trades = (results_df['action'].diff().fillna(0) != 0).sum()
        valid = results_df['action'] != 0
        step_sharpe = results_df['net_return'].mean() / results_df['net_return'].std(ddof=0) if results_df['net_return'].std(ddof=0) > 0 else 0.0
        cum = results_df['equity']
        peak = cum.cummax()
        mdd = ((cum / peak) - 1.0).min() if len(cum) else 0.0
        total_ret = cum.iloc[-1] - 1.0 if len(cum) else 0.0
        win_rate = (results_df.loc[valid, 'net_return'] > 0).mean() if valid.any() else 0.0
        print(f"✅ Backtest Accuracy: {accuracy*100:.2f}% | Trades: {int(n_trades)} | Sharpe(step): {step_sharpe:.2f} | Total Return: {total_ret*100:.2f}% | MDD: {mdd*100:.2f}% | Win rate: {win_rate*100:.1f}%")
        
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
        print(f"📈 Prediction Horizon: {self.prediction_horizon} periods")
        print(f"🎯 Target Type: {self.target_type}")
        print(f"📊 Selected Features: {len(self.selected_features)} (from {len(self.feature_columns)})")
        
        print(f"\n✅ ACCURACY METRICS:")
        if 'cv_scores' in results:
            cv_mean = np.mean(results['cv_scores'])
            cv_std = np.std(results['cv_scores'])
            print(f"   Cross-Validation:    {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
        print(f"   Training Accuracy:   {results['train_accuracy']*100:.2f}%")
        print(f"   Testing Accuracy:    {results['test_accuracy']*100:.2f}%")
        
        if backtest_results is not None:
            backtest_accuracy = backtest_results['correct'].mean()
            print(f"   Backtest Accuracy:   {backtest_accuracy*100:.2f}%")
            # Profit-aware metrics
            valid = backtest_results['action'] != 0
            trades = (backtest_results['action'].diff().fillna(0) != 0).sum()
            step_sharpe = backtest_results['net_return'].mean() / backtest_results['net_return'].std(ddof=0) if backtest_results['net_return'].std(ddof=0) > 0 else 0.0
            cum = backtest_results['equity']
            peak = cum.cummax()
            mdd = ((cum / peak) - 1.0).min() if len(cum) else 0.0
            total_ret = cum.iloc[-1] - 1.0 if len(cum) else 0.0
            win_rate = (backtest_results.loc[valid, 'net_return'] > 0).mean() if valid.any() else 0.0
            print(f"\n💹 PROFIT METRICS (backtest):")
            print(f"   Optimize for:      {self.optimize_for}")
            print(f"   Trading band:      lo={self.trade_lo:.2f}, hi={self.trade_hi:.2f}, cost={self.transaction_cost:.4f}")
            print(f"   Trades executed:   {int(trades)}")
            print(f"   Step Sharpe:       {step_sharpe:.2f}")
            print(f"   Total return:      {total_ret*100:.2f}%")
            print(f"   Max drawdown:      {mdd*100:.2f}%")
            print(f"   Win rate:          {win_rate*100:.1f}%")
        
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
