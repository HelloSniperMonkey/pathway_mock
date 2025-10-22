"""
Support NLP Engine Module
Processes user queries using LLM and generates responses
"""

from typing import Dict, Optional, List, Tuple
import json
from datetime import datetime

# Try to import various LLM libraries
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class SupportNLPEngine:
    """NLP engine for customer support queries"""

    # Financial keywords for better intent detection
    FINANCIAL_KEYWORDS = {
        "investment": ["invest", "portfolio", "return", "asset", "market"],
        "trading": ["trade", "buy", "sell", "stock", "crypto", "forex"],
        "loan": ["loan", "borrow", "credit", "interest", "payment"],
        "savings": ["save", "deposit", "interest", "account", "compound"],
        "insurance": ["insurance", "policy", "coverage", "claim", "premium"],
        "tax": ["tax", "deduction", "filing", "return", "irs"],
        "cryptocurrency": ["bitcoin", "ethereum", "crypto", "blockchain", "wallet"],
        "general": ["help", "info", "question", "how", "what", "when"]
    }

    # Pre-trained responses for common queries
    FAQ_RESPONSES = {
        "investment_basics": {
            "keywords": ["invest", "start", "beginner", "how", "first time"],
            "response": (
                "To start investing:\n"
                "1. Define your goals and risk tolerance\n"
                "2. Start with low-cost index funds or ETFs\n"
                "3. Diversify your portfolio across assets\n"
                "4. Invest consistently (Dollar-Cost Averaging)\n"
                "5. Review and rebalance periodically\n"
                "Would you like specific advice on any of these?"
            )
        },
        "crypto_risk": {
            "keywords": ["crypto", "risk", "dangerous", "safe"],
            "response": (
                "Cryptocurrency carries significant risks:\n"
                "⚠️  High volatility and price fluctuations\n"
                "⚠️  Regulatory uncertainty\n"
                "⚠️  Security and hacking risks\n"
                "⚠️  Market manipulation\n\n"
                "Recommendations:\n"
                "✓ Only invest what you can afford to lose\n"
                "✓ Use reputable exchanges\n"
                "✓ Enable 2FA security\n"
                "✓ Start with small amounts"
            )
        },
        "emergency_fund": {
            "keywords": ["emergency", "fund", "savings", "backup"],
            "response": (
                "Building an emergency fund:\n"
                "• Aim for 3-6 months of living expenses\n"
                "• Keep it in liquid, easily accessible accounts\n"
                "• Use high-yield savings accounts for growth\n"
                "• Build it gradually if needed\n"
                "• Don't touch it except for emergencies\n\n"
                "This provides financial security and peace of mind."
            )
        },
        "market_crash": {
            "keywords": ["crash", "market", "down", "fall", "panic"],
            "response": (
                "During market downturns:\n"
                "✓ Don't panic sell - historically, markets recover\n"
                "✓ Consider dollar-cost averaging\n"
                "✓ Review your asset allocation\n"
                "✓ Use it as buying opportunity if you have cash\n"
                "✓ Focus on long-term goals, not short-term noise\n\n"
                "Remember: Time in market beats timing the market."
            )
        }
    }

    def __init__(self, use_gemini: bool = False, api_key: Optional[str] = None):
        """
        Initialize NLP engine
        
        Args:
            use_gemini: Whether to use Google Gemini API
            api_key: Google Gemini API key (if using Gemini)
        """
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.use_transformers = TRANSFORMERS_AVAILABLE and not use_gemini

        if use_gemini and GEMINI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)

        # Initialize local model if available
        self.qa_pipeline = None
        if self.use_transformers:
            try:
                self.qa_pipeline = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small"
                )
                print("✓ Transformer model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                self.use_transformers = False

    def detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect the intent of user query
        
        Args:
            query: User query
            
        Returns:
            (intent_category, confidence)
        """
        query_lower = query.lower()
        intent_scores = {}

        # Score each category
        for category, keywords in self.FINANCIAL_KEYWORDS.items():
            score = sum(
                query_lower.count(keyword)
                for keyword in keywords
            )
            intent_scores[category] = score

        # Find highest scoring category
        if max(intent_scores.values()) > 0:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(1.0, intent_scores[best_intent] / 3)
            return best_intent, confidence
        else:
            return "general", 0.5

    def check_faq(self, query: str) -> Optional[str]:
        """
        Check if query matches FAQ responses
        
        Args:
            query: User query
            
        Returns:
            FAQ response if found, None otherwise
        """
        query_lower = query.lower()

        for faq_key, faq_data in self.FAQ_RESPONSES.items():
            keywords = faq_data["keywords"]
            if any(keyword in query_lower for keyword in keywords):
                return faq_data["response"]

        return None

    def generate_response_gemini(self, query: str, context: str = "") -> str:
        """
        Generate response using Google Gemini API
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Generated response
        """
        if not GEMINI_AVAILABLE:
            return self.generate_response_fallback(query, context)

        try:
            system_prompt = (
                "You are a helpful financial advisor AI assistant. "
                "Provide accurate, practical financial advice in a clear and concise manner. "
                "Disclaimer: This is general financial information, not professional advice."
            )

            full_prompt = f"{system_prompt}\n\n"
            
            if context:
                full_prompt += f"Context: {context}\n\n"
            
            full_prompt += f"User Question: {query}\n\nAnswer:"

            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(full_prompt)

            return response.text

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return self.generate_response_fallback(query, context)

    def generate_response_transformer(self, query: str, context: str = "") -> str:
        """
        Generate response using transformer model
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Generated response
        """
        if not self.use_transformers:
            return self.generate_response_fallback(query, context)

        try:
            prompt = f"Answer this financial question concisely: {query}"
            if context:
                prompt = f"Context: {context}\n" + prompt

            response = self.qa_pipeline(prompt, max_length=150)
            return response[0]["generated_text"]

        except Exception as e:
            print(f"Error generating response with transformer: {e}")
            return self.generate_response_fallback(query, context)

    def generate_response_fallback(self, query: str, context: str = "") -> str:
        """
        Generate response using fallback strategy
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Generated response
        """
        intent, confidence = self.detect_intent(query)

        # Build response based on intent
        response_parts = []

        if confidence > 0.7:
            response_parts.append(f"Based on your interest in {intent}:")
            response_parts.append("")

        # Add FAQ response if available
        faq_response = self.check_faq(query)
        if faq_response:
            response_parts.append(faq_response)
        else:
            # Generic response
            response_parts.append(
                f"Thank you for your question about {intent}. "
                "Here are some general principles:\n"
            )

            if intent == "investment":
                response_parts.append("• Start with a clear investment strategy")
                response_parts.append("• Diversify across different asset classes")
                response_parts.append("• Monitor your portfolio regularly")
            elif intent == "trading":
                response_parts.append("• Develop a trading plan before entering trades")
                response_parts.append("• Use stop-loss orders for risk management")
                response_parts.append("• Never risk more than you can afford to lose")
            elif intent == "cryptocurrency":
                response_parts.append("• Understand the technology and risks")
                response_parts.append("• Start with small investments")
                response_parts.append("• Use secure wallets and exchanges")
            else:
                response_parts.append("• Consider consulting a financial professional")
                response_parts.append("• Do thorough research before making decisions")

        return "\n".join(response_parts)

    def process_query(
        self,
        query: str,
        context: str = "",
        use_api: bool = False
    ) -> Dict[str, str]:
        """
        Process user query and generate response
        
        Args:
            query: User query
            context: Additional context
            use_api: Whether to use API (Gemini)
            
        Returns:
            Dictionary with response and metadata
        """
        # Check FAQ first
        faq_response = self.check_faq(query)
        if faq_response:
            return {
                "response": faq_response,
                "source": "faq",
                "intent": self.detect_intent(query)[0],
                "timestamp": datetime.now().isoformat()
            }

        # Generate response
        if use_api and self.use_gemini:
            response = self.generate_response_gemini(query, context)
        elif self.use_transformers:
            response = self.generate_response_transformer(query, context)
        else:
            response = self.generate_response_fallback(query, context)

        return {
            "response": response,
            "source": "llm",
            "intent": self.detect_intent(query)[0],
            "timestamp": datetime.now().isoformat()
        }

    def validate_response(self, query: str, response: str) -> Dict[str, bool]:
        """
        Validate response quality
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            Dictionary with validation results
        """
        validations = {
            "non_empty": len(response.strip()) > 0,
            "relevant": any(
                keyword in response.lower()
                for keyword in query.lower().split()
                if len(keyword) > 3
            ),
            "appropriate_length": 50 < len(response) < 2000,
            "contains_disclaimer": any(
                disclaimer in response.lower()
                for disclaimer in ["not financial advice", "consult", "professional"]
            )
        }

        return validations
