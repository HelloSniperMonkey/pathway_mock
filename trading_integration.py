"""
Integration Module: Connect AI Trading to Main Financial Assistant
"""

from typing import Dict, Any
import os


class TradingAIIntegration:
    """
    Integrates AI price prediction with the main Financial AI Assistant.
    Provides a clean interface to run trading predictions from the main system.
    """
    
    def __init__(self):
        """Initialize the trading AI integration"""
        self.predictor = None
        self.last_results = None
        
    def run_price_prediction_analysis(self, csv_file_path: str, 
                                      model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Run complete AI price prediction analysis.
        
        Args:
            csv_file_path: Path to CSV with historical data
            model_type: 'random_forest' or 'gradient_boosting'
            
        Returns:
            Dictionary with results and file paths
        """
        try:
            from integrated_ai_trading import run_ai_prediction_pipeline
            
            print("\n" + "="*70)
            print("🤖 AI PRICE PREDICTION MODULE")
            print("="*70)
            
            # Run the complete pipeline
            predictor, results, backtest_results = run_ai_prediction_pipeline(
                csv_file_path,
                model_type=model_type
            )
            
            # Store for later use
            self.predictor = predictor
            self.last_results = {
                'training_results': results,
                'backtest_results': backtest_results,
                'model_type': model_type,
                'data_file': csv_file_path
            }
            
            return {
                'success': True,
                'train_accuracy': results['train_accuracy'],
                'test_accuracy': results['test_accuracy'],
                'backtest_accuracy': backtest_results['correct'].mean(),
                'plots': ['model_performance.png', 'price_predictions.png'],
                'csv': 'prediction_results.csv',
                'message': 'AI prediction analysis complete!'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to run AI prediction analysis'
            }
    
    def get_prediction_summary(self) -> str:
        """
        Get a summary of the last prediction run.
        
        Returns:
            Formatted string with key metrics
        """
        if not self.last_results:
            return "No predictions have been run yet."
        
        results = self.last_results['training_results']
        backtest = self.last_results['backtest_results']
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║         AI PRICE PREDICTION - RESULTS SUMMARY                 ║
╚══════════════════════════════════════════════════════════════╝

📊 Model Type: {self.last_results['model_type'].upper()}
📁 Data File: {os.path.basename(self.last_results['data_file'])}

🎯 ACCURACY METRICS:
   Training:   {results['train_accuracy']*100:.2f}%
   Testing:    {results['test_accuracy']*100:.2f}%
   Backtest:   {backtest['correct'].mean()*100:.2f}%

🔝 TOP 5 IMPORTANT FEATURES:
"""
        for idx, row in results['feature_importance'].head(5).iterrows():
            summary += f"   {idx+1}. {row['feature']:20s} {row['importance']:.4f}\n"
        
        summary += f"""
📈 PREDICTION DISTRIBUTION:
   Up Predictions:   {(backtest['predicted_direction'] == 1).sum()}
   Down Predictions: {(backtest['predicted_direction'] == 0).sum()}

💪 Average Confidence: {backtest['confidence'].mean()*100:.2f}%

✅ Status: Ready for deployment!
"""
        return summary
    
    def predict_next_move(self, csv_file_path: str = None) -> Dict[str, Any]:
        """
        Predict the next price movement.
        
        Args:
            csv_file_path: Optional new data file
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.predictor:
            return {
                'success': False,
                'message': 'Model not trained. Run analysis first.'
            }
        
        try:
            if csv_file_path:
                from integrated_ai_trading import (
                    read_csv_to_dataframe, 
                    calculate_all_indicators
                )
                df = read_csv_to_dataframe(csv_file_path)
                df = calculate_all_indicators(df)
            else:
                # Use last data
                return {
                    'success': False,
                    'message': 'No data provided for prediction'
                }
            
            # Predict last candle
            current_idx = len(df) - 2  # Second to last
            prediction, confidence = self.predictor.predict_next(df, current_idx)
            
            direction = "UP 📈" if prediction == 1 else "DOWN 📉"
            
            return {
                'success': True,
                'direction': direction,
                'confidence': confidence * 100,
                'timestamp': df.index[current_idx],
                'current_price': df.iloc[current_idx]['close'],
                'message': f'Prediction: {direction} with {confidence*100:.1f}% confidence'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Prediction failed'
            }


# Integration with main.py
def add_to_financial_assistant():
    """
    Example of how to integrate with main Financial AI Assistant.
    Add this to your main.py file.
    """
    integration_code = '''
# In main.py, add this import at the top:
from trading_integration import TradingAIIntegration

# In FinancialAIAssistant.__init__(), add:
self.trading_ai = TradingAIIntegration()

# Add this method to FinancialAIAssistant class:
def run_trading_analysis(self, csv_file: str):
    """Run AI-powered trading analysis"""
    print("\\n🤖 Starting AI price prediction analysis...")
    results = self.trading_ai.run_price_prediction_analysis(csv_file)
    
    if results['success']:
        print(f"\\n✅ Analysis complete!")
        print(f"   Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"   Backtest Accuracy: {results['backtest_accuracy']*100:.2f}%")
        print(f"\\n📊 Generated plots:")
        for plot in results['plots']:
            print(f"   - {plot}")
        print(f"\\n📁 Detailed results: {results['csv']}")
        
        # Show summary
        print(self.trading_ai.get_prediction_summary())
    else:
        print(f"\\n❌ Analysis failed: {results['message']}")

# Add to interactive menu:
def interactive_menu(self):
    """Enhanced menu with trading analysis"""
    while True:
        print("\\n" + "="*70)
        print("FINANCIAL AI ASSISTANT - MAIN MENU")
        print("="*70)
        print("1. Run KYC Verification")
        print("2. Customer Support Session")
        print("3. 🆕 AI Trading Analysis")  # NEW OPTION
        print("4. Exit")
        
        choice = input("\\nSelect option: ").strip()
        
        if choice == '3':
            csv_file = input("Enter path to CSV file: ").strip()
            if csv_file:
                self.run_trading_analysis(csv_file)
            else:
                print("❌ No file path provided")
    '''
    return integration_code


def main():
    """Demonstration of trading AI integration"""
    print("="*70)
    print("🔧 TRADING AI INTEGRATION MODULE")
    print("="*70)
    
    print("\n📝 This module provides integration with the main Financial AI Assistant.")
    print("\nTo integrate:")
    print("1. Import TradingAIIntegration in main.py")
    print("2. Add trading_ai instance to FinancialAIAssistant")
    print("3. Add run_trading_analysis() method")
    print("4. Update interactive menu with option 3")
    
    print("\n✅ Integration code is ready to use!")
    print("\nExample usage:")
    print("""
    from trading_integration import TradingAIIntegration
    
    # Initialize
    trading = TradingAIIntegration()
    
    # Run analysis
    results = trading.run_price_prediction_analysis('Bitcoin_data.csv')
    
    # Get summary
    print(trading.get_prediction_summary())
    
    # Make prediction
    pred = trading.predict_next_move('Bitcoin_data.csv')
    print(f"Next move: {pred['direction']} ({pred['confidence']:.1f}%)")
    """)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
