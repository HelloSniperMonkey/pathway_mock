"""
Support Response Generator Module
Generates contextual responses based on user profile and query
Bonus feature: Personalized responses based on history
"""

from typing import Dict, List, Optional
from datetime import datetime
from support_nlp_engine import SupportNLPEngine
from support_user_profiler import UserProfile


class SupportResponseGenerator:
    """Generates contextual responses for customer support"""

    # Response templates based on profile characteristics
    PROFILE_TEMPLATES = {
        "beginner": {
            "greeting": "Welcome! I see you're new to investing. Let me help you get started.",
            "tone": "educational"
        },
        "experienced": {
            "greeting": "Great to see you again! Based on your investment history,",
            "tone": "professional"
        },
        "conservative": {
            "greeting": "I understand you prefer safe, stable investments.",
            "tone": "cautious"
        },
        "aggressive": {
            "greeting": "I see you're interested in growth-focused investments.",
            "tone": "dynamic"
        }
    }

    def __init__(self, nlp_engine: Optional[SupportNLPEngine] = None):
        """
        Initialize response generator
        
        Args:
            nlp_engine: SupportNLPEngine instance for base responses
        """
        self.nlp_engine = nlp_engine or SupportNLPEngine()

    def determine_profile_level(self, profile: UserProfile) -> str:
        """
        Determine user expertise level
        
        Args:
            profile: User profile
            
        Returns:
            Expertise level (beginner, intermediate, experienced)
        """
        if len(profile.conversation_history) < 3:
            return "beginner"
        elif len(profile.conversation_history) < 10:
            return "intermediate"
        else:
            return "experienced"

    def generate_personalized_greeting(self, profile: UserProfile) -> str:
        """
        Generate personalized greeting based on profile
        
        Args:
            profile: User profile
            
        Returns:
            Personalized greeting
        """
        greeting_parts = []

        # Greeting with name
        if profile.name:
            greeting_parts.append(f"Hello {profile.name}!")
        else:
            greeting_parts.append("Hello!")

        # Context based on risk tolerance
        if profile.risk_tolerance:
            template = self.PROFILE_TEMPLATES.get(
                profile.risk_tolerance,
                self.PROFILE_TEMPLATES["conservative"]
            )
            greeting_parts.append(template["greeting"])

        # Context based on interests
        if profile.investment_interests:
            interests_str = ", ".join(profile.investment_interests[:2])
            greeting_parts.append(f"I see you're interested in {interests_str}.")

        return " ".join(greeting_parts)

    def customize_nlp_response(
        self,
        base_response: str,
        profile: UserProfile
    ) -> str:
        """
        Customize NLP response based on user profile
        
        Args:
            base_response: Base response from NLP engine
            profile: User profile
            
        Returns:
            Customized response
        """
        customized = []

        # Add personalized greeting if it's the first response
        if len(profile.conversation_history) == 1:
            customized.append(self.generate_personalized_greeting(profile))
            customized.append("")

        # Add profile-specific recommendations
        if profile.risk_tolerance == "conservative":
            customized.append("📌 Based on your conservative approach, I recommend:")
            customized.append("• Focus on stable, dividend-paying stocks")
            customized.append("• Consider bonds and fixed income securities")
            customized.append("• Maintain a 60/40 stock-to-bond ratio")
        elif profile.risk_tolerance == "aggressive":
            customized.append("📌 Based on your growth focus, consider:")
            customized.append("• Growth stocks in emerging sectors")
            customized.append("• Technology and innovation funds")
            customized.append("• Crypto allocation (small % of portfolio)")

        customized.append("")
        customized.append("Base Advice:")
        customized.append(base_response)

        # Add goal-specific advice
        if profile.financial_goals:
            customized.append("")
            customized.append("Regarding your goals:")
            for goal in profile.financial_goals[:2]:
                customized.append(f"• {goal.replace('_', ' ').title()}: Consider specific strategies for this.")

        return "\n".join(customized)

    def generate_follow_up_questions(self, profile: UserProfile, query: str) -> List[str]:
        """
        Generate relevant follow-up questions
        
        Args:
            profile: User profile
            query: Original query
            
        Returns:
            List of follow-up questions
        """
        follow_ups = []

        # If investment amount is unknown
        if profile.extracted_entities.get("investment_amount") == "Unknown":
            follow_ups.append("How much are you looking to invest?")

        # If goals not specified
        if not profile.financial_goals:
            follow_ups.append("What are your main financial goals?")

        # If timeline not mentioned
        if len(profile.conversation_history) == 1:
            follow_ups.append("What is your investment timeline?")

        # If risk tolerance not detected
        if not profile.risk_tolerance:
            follow_ups.append("How would you rate your risk tolerance?")

        # Interest-specific questions
        if "crypto" in profile.investment_interests:
            follow_ups.append("How experienced are you with cryptocurrency?")

        if "trading" in profile.investment_interests:
            follow_ups.append("Are you interested in day trading or long-term investing?")

        return follow_ups[:3]  # Return top 3 questions

    def generate_contextual_response(
        self,
        query: str,
        profile: UserProfile,
        base_response: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive contextual response
        
        Args:
            query: User query
            profile: User profile
            base_response: Optional base response to customize
            
        Returns:
            Dictionary with response and metadata
        """
        # Get base response if not provided
        if not base_response:
            nlp_result = self.nlp_engine.process_query(query)
            base_response = nlp_result["response"]

        # Customize response based on profile
        customized_response = self.customize_nlp_response(base_response, profile)

        # Generate follow-up questions
        follow_ups = self.generate_follow_up_questions(profile, query)

        return {
            "response": customized_response,
            "follow_up_questions": "\n".join(
                f"• {q}" for q in follow_ups
            ),
            "expertise_level": self.determine_profile_level(profile),
            "timestamp": datetime.now().isoformat()
        }

    def generate_summary_report(self, profile: UserProfile) -> str:
        """
        Generate comprehensive user interaction report
        
        Args:
            profile: User profile
            
        Returns:
            Report as string
        """
        report = []

        report.append("=" * 70)
        report.append("CUSTOMER INTERACTION SUMMARY REPORT")
        report.append("=" * 70)
        report.append("")

        # User information
        report.append("USER INFORMATION:")
        report.append(f"  User ID: {profile.user_id}")
        if profile.name:
            report.append(f"  Name: {profile.name}")
        if profile.email:
            report.append(f"  Email: {profile.email}")
        if profile.location:
            report.append(f"  Location: {profile.location}")
        report.append("")

        # Investment Profile
        report.append("💼 INVESTMENT PROFILE:")
        report.append(f"  Risk Tolerance: {profile.risk_tolerance or 'Not determined'}")
        if profile.investment_interests:
            report.append(f"  Interests: {', '.join(profile.investment_interests)}")
        if profile.financial_goals:
            report.append(f"  Goals: {', '.join(g.replace('_', ' ') for g in profile.financial_goals)}")
        report.append("")

        # Interaction History
        report.append("📞 INTERACTION HISTORY:")
        report.append(f"  Total Conversations: {len(profile.conversation_history)}")
        report.append(f"  Profile Created: {profile.created_at}")
        report.append(f"  Last Updated: {profile.updated_at}")
        report.append("")

        # Recent Queries
        if profile.conversation_history:
            report.append("  Recent Queries:")
            for i, conv in enumerate(profile.conversation_history[-3:], 1):
                query_preview = conv["query"][:50] + "..." if len(conv["query"]) > 50 else conv["query"]
                report.append(f"    {i}. {query_preview}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        expertise = self.determine_profile_level(profile)
        if expertise == "beginner":
            report.append("  • Provide educational resources and guides")
            report.append("  • Simplify complex financial concepts")
            report.append("  • Focus on fundamentals and risk management")
        elif expertise == "experienced":
            report.append("  • Discuss advanced strategies")
            report.append("  • Provide market analysis and insights")
            report.append("  • Consider portfolio optimization")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)
