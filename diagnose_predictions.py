"""
Diagnostic script to understand prediction behavior
"""
import pandas as pd
import numpy as np

# Load prediction results
df = pd.read_csv('prediction_results.csv')

print("="*70)
print("PREDICTION ANALYSIS")
print("="*70)

print("\n1. PREDICTION DISTRIBUTION:")
print(df['predicted_direction'].value_counts())
print(f"\nUp predictions: {(df['predicted_direction']==1).sum()} ({(df['predicted_direction']==1).mean()*100:.1f}%)")
print(f"Down predictions: {(df['predicted_direction']==0).sum()} ({(df['predicted_direction']==0).mean()*100:.1f}%)")

print("\n2. ACTUAL DIRECTION DISTRIBUTION:")
print(df['actual_direction'].value_counts())
print(f"\nActual Up: {(df['actual_direction']==1).sum()} ({(df['actual_direction']==1).mean()*100:.1f}%)")
print(f"Actual Down: {(df['actual_direction']==0).sum()} ({(df['actual_direction']==0).mean()*100:.1f}%)")

print("\n3. CONFIDENCE STATISTICS:")
print(df['confidence'].describe())

print("\n4. ACCURACY BREAKDOWN:")
print(f"Overall Accuracy: {df['correct'].mean()*100:.2f}%")
print(f"When predicting UP: {df[df['predicted_direction']==1]['correct'].mean()*100:.2f}%")
print(f"When predicting DOWN: {df[df['predicted_direction']==0]['correct'].mean()*100:.2f}%")

print("\n5. CONFIDENCE vs ACCURACY:")
high_conf = df[df['confidence'] > 0.6]
low_conf = df[df['confidence'] <= 0.6]
print(f"High confidence (>60%): {len(high_conf)} predictions, {high_conf['correct'].mean()*100:.2f}% accurate")
print(f"Low confidence (≤60%): {len(low_conf)} predictions, {low_conf['correct'].mean()*100:.2f}% accurate")

print("\n6. PRICE CHANGE ANALYSIS:")
df['price_change_pct'] = ((df['next_price'] - df['current_price']) / df['current_price']) * 100
print(f"Average price change: {df['price_change_pct'].mean():.3f}%")
print(f"Std dev of price change: {df['price_change_pct'].std():.3f}%")
print(f"Max price increase: {df['price_change_pct'].max():.2f}%")
print(f"Max price decrease: {df['price_change_pct'].min():.2f}%")

print("\n7. POTENTIAL ISSUE DETECTION:")
# Check if model is just predicting one class most of the time
pred_imbalance = abs(df['predicted_direction'].mean() - 0.5)
if pred_imbalance > 0.3:
    print("⚠️  WARNING: Model is heavily biased toward one direction!")
elif pred_imbalance > 0.15:
    print("⚠️  CAUTION: Model shows bias toward one direction")
else:
    print("✓ Predictions are reasonably balanced")

# Check if confidence is meaningful
avg_conf = df['confidence'].mean()
if avg_conf < 0.55:
    print("⚠️  WARNING: Very low average confidence - model is mostly guessing!")
elif avg_conf < 0.65:
    print("⚠️  CAUTION: Low confidence - model has weak signal")
else:
    print("✓ Confidence levels suggest model has learned something")

# Check if accuracy is better than random
accuracy = df['correct'].mean()
if accuracy < 0.52:
    print("⚠️  WARNING: Accuracy near random (50%) - model not learning!")
elif accuracy < 0.58:
    print("⚠️  CAUTION: Accuracy only slightly better than random")
else:
    print("✓ Accuracy is meaningfully above random chance")

print("\n" + "="*70)
print("RECOMMENDATION:")
if accuracy < 0.55 and avg_conf < 0.60:
    print("🔴 Model is not learning meaningful patterns!")
    print("\nPossible causes:")
    print("  1. Features don't have predictive power for daily price changes")
    print("  2. Target is too noisy (daily movements are hard to predict)")
    print("  3. Model parameters need tuning")
    print("  4. Need more training data")
    print("\nSuggestions:")
    print("  • Try predicting longer timeframes (e.g., 4-hour or 8-hour candles)")
    print("  • Add more features or feature engineering")
    print("  • Try different model architectures (LSTM, etc.)")
    print("  • Consider ensemble methods")
elif accuracy > 0.55:
    print("🟡 Model shows some learning but could be improved")
else:
    print("🟢 Model performance looks reasonable")
