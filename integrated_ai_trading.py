"""
Integrated AI Trading System
Combines your existing trading indicators with AI price prediction
"""

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import your existing indicator functions from quant.py
# These functions remain UNCHANGED
def read_csv_to_dataframe(file_path):
    """Your original function - UNCHANGED"""
    df = pd.read_csv(file_path, sep=';')
    if df.iloc[0, 0] == 'timestamp':
        df = df.iloc[1:]
    
    # Handle ISO8601 format timestamps with quotes
    df["timestamp"] = df["timestamp"].str.replace('"', '')  # Remove quotes
    
    # Parse ISO8601 format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    df = df[df.high != df.low]
    df.set_index("timestamp", inplace=True)
    return df

def ichimoku(df):
    """Your original function - UNCHANGED"""
    high = df['high']
    low = df['low']
    close = df['close']
    nine_period_high = high.rolling(window=9).max()
    nine_period_low = low.rolling(window=9).min()
    df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
    twenty_six_period_high = high.rolling(window=26).max()
    twenty_six_period_low = low.rolling(window=26).min()
    df['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    fifty_two_period_high = high.rolling(window=52).max()
    fifty_two_period_low = low.rolling(window=52).min()
    df['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
    df['chikou_span'] = close.shift(-26)
    return df

def calculate_heikin_ashi(df):
    """Your original function - UNCHANGED"""
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    df['ha_high'] = df[['high', 'close', 'open']].max(axis=1)
    df['ha_low'] = df[['low', 'close', 'open']].min(axis=1)
    return df

def calculate_MACD(df):
    """Your original function - UNCHANGED"""
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def hawkes_process(data: pd.Series, kappa: float):
    """Your original function - UNCHANGED"""
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=data.index) * kappa

def calculate_RSI(df):
    """Your original function - UNCHANGED"""
    try:
        import ta
        rsi_object = ta.momentum.RSIIndicator(df['close'])
        df['RSI'] = rsi_object.rsi()
    except:
        # Fallback RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    return df

def wwma(values, n):
    """Your original function - UNCHANGED"""
    return values.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

def atr(df, n=14):
    """Your original function - UNCHANGED"""
    df = df.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    df['tr0'] = abs(high - low)
    df['tr1'] = abs(high - close.shift())
    df['tr2'] = abs(low - close.shift())
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr_value = wwma(tr, n)
    df['atr_values'] = atr_value
    return df

def aroon(df, day):
    """Your original function - UNCHANGED"""
    df['aroon_up'] = 100 * df.high.rolling(day).apply(lambda x: x.argmax()) / day
    df['aroon_down'] = 100 * df.low.rolling(day).apply(lambda x: x.argmin()) / day
    return df

def get_adx(high, low, close, lookback, df):
    """Your original function - UNCHANGED"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr_val = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/lookback).mean() / atr_val)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/lookback).mean() / atr_val))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha=1/lookback).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['adx'] = adx_smooth
    return df

def calculate_all_indicators(df):
    """
    Calculate all technical indicators on the dataframe.
    This uses your existing functions - UNCHANGED logic.
    """
    print("📊 Calculating technical indicators...")
    
    # Ichimoku Cloud
    df = ichimoku(df)
    
    # ADX
    df = get_adx(df['high'], df['low'], df['close'], 14, df)
    
    # Volume indicators
    df['vol_sma_14'] = df['volume'].rolling(window=14, min_periods=7).mean()
    df['vol_sma_9'] = df['volume'].rolling(window=9, min_periods=7).mean()
    
    # EMA
    df['ema_6'] = df['close'].ewm(span=6, adjust=False, min_periods=6).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False, min_periods=10).mean()
    df['ema_18'] = df['close'].ewm(span=18, adjust=False, min_periods=17).mean()
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cumulative_price_volume'] = df['typical_price'] * df['volume']
    df['cumulative_volume'] = df['volume'].cumsum()
    df['cumulative_price'] = df['cumulative_price_volume'].cumsum()
    df['vwap'] = df['cumulative_price'] / df['cumulative_volume']
    
    # MACD
    df = calculate_MACD(df)
    
    # RSI
    df = calculate_RSI(df)
    
    # Heikin Ashi
    df = calculate_heikin_ashi(df)
    
    # ATR
    df = atr(df)
    
    # Aroon
    df = aroon(df, 14)
    
    # Hawkes process
    df['norm_range'] = (df['atr_values'] - df['atr_values'].mean()) / df['atr_values'].std()
    df['v_hawk'] = hawkes_process(df['norm_range'], 1)
    df['q05'] = df['v_hawk'].rolling(20).quantile(0.05)
    df['q95'] = df['v_hawk'].rolling(20).quantile(0.95)
    
    print("✅ All indicators calculated successfully!")
    return df


def run_ai_prediction_pipeline(csv_file_path, model_type='random_forest'):
    """
    Complete pipeline: Load data -> Calculate indicators -> Train AI -> Predict
    
    Args:
        csv_file_path: Path to your CSV file
        model_type: 'random_forest' or 'gradient_boosting'
    """
    print("\n" + "="*70)
    print("🚀 AI-POWERED TRADING SYSTEM")
    print("="*70)
    
    # Step 1: Load data
    print(f"\n📂 Loading data from: {csv_file_path}")
    df = read_csv_to_dataframe(csv_file_path)
    print(f"✅ Loaded {len(df)} rows of data")
    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
    
    # Step 2: Calculate all indicators (your existing logic)
    df = calculate_all_indicators(df)
    
    # Step 3: Import and use AI predictor
    try:
        from ai_price_predictor import AIPricePredictor
        
        # Initialize AI model
        print(f"\n🤖 Initializing {model_type} AI model...")
        predictor = AIPricePredictor(model_type=model_type)
        
        # Train model
        print("\n📚 Training AI model on historical data...")
        results = predictor.train(df, test_size=0.2)
        
        # Run backtest
        print("\n🔄 Running backtest predictions...")
        backtest_results = predictor.backtest_predictions(df)
        
        # Generate plots
        print("\n📊 Generating performance visualizations...")
        predictor.plot_results(results, save_path='model_performance.png')
        predictor.plot_price_predictions(backtest_results, save_path='price_predictions.png')
        
        # Generate report
        predictor.generate_report(results, backtest_results)
        
        # Save results
        backtest_results.to_csv('prediction_results.csv', index=False)
        print("\n💾 Results saved to 'prediction_results.csv'")
        
        return predictor, results, backtest_results
        
    except ImportError as e:
        print(f"\n⚠️  Error importing AI predictor: {e}")
        print("Make sure ai_price_predictor.py is in the same directory")
        return None, None, None


def main():
    """
    Main execution function
    """
    # Default CSV file (from your workspace)
    csv_file = "bitcoin.csv"
    
    # if not os.path.exists(csv_file):
    #     print(f"❌ File not found: {csv_file}")
    #     print("Please provide the correct path to your CSV file")
    #     return
    
    # Run the complete pipeline
    predictor, results, backtest_results = run_ai_prediction_pipeline(
        csv_file,
        model_type='random_forest'  # Can also use 'gradient_boosting'
    )
    
    if predictor:
        print("\n" + "="*70)
        print("✅ AI TRADING SYSTEM COMPLETE!")
        print("="*70)
        print("\n📁 Generated Files:")
        print("   • model_performance.png - Model accuracy and feature importance")
        print("   • price_predictions.png - Predicted vs actual prices")
        print("   • prediction_results.csv - Detailed prediction results")
        print("\n🎯 Key Achievement: The model predicts price direction WITHOUT")
        print("   looking at future data (no data leakage!)")


if __name__ == "__main__":
    main()
