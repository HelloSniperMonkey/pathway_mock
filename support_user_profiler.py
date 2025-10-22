"""
Support User Profiler Module
Extracts and stores user information from conversation queries
Bonus feature: Builds user profile over time
"""

import re
import json
from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class UserProfile:
    """User profile from conversation history"""
    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    age_group: Optional[str] = None
    investment_interests: List[str] = None
    risk_tolerance: Optional[str] = None
    location: Optional[str] = None
    financial_goals: List[str] = None
    extracted_entities: Dict[str, str] = None
    conversation_history: List[Dict] = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if self.investment_interests is None:
            self.investment_interests = []
        if self.financial_goals is None:
            self.financial_goals = []
        if self.extracted_entities is None:
            self.extracted_entities = {}
        if self.conversation_history is None:
            self.conversation_history = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


class SupportUserProfiler:
    """Extracts user information from queries and builds profiles"""

    # Patterns for entity extraction
    PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\+?1?\d{9,15}",
        "age": r"\b([1-9]|[1-9][0-9]|1[01]\d)\s*(?:years?|yo|yrs?|age)\b",
        "investment_amount": r"\$?\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:k|K|million|M))?"
    }

    # Keywords for intent extraction
    INTENT_KEYWORDS = {
        "stocks": ["stock", "share", "equity", "sec", "nyse"],
        "crypto": ["bitcoin", "ethereum", "crypto", "blockchain", "defi"],
        "real_estate": ["property", "real estate", "house", "mortgage"],
        "bonds": ["bond", "treasury", "fixed income"],
        "mutual_funds": ["fund", "etf", "mutual"],
        "trading": ["day trade", "swing trade", "short", "long position"],
        "retirement": ["retirement", "401k", "ira", "pension"],
        "savings": ["save", "savings", "deposit", "high yield"]
    }

    # Risk tolerance keywords
    RISK_KEYWORDS = {
        "conservative": ["conservative", "safe", "low risk", "stable", "capital preservation"],
        "moderate": ["balanced", "moderate", "medium risk", "diversified"],
        "aggressive": ["aggressive", "high risk", "growth", "volatile", "risk-taking"]
    }

    def __init__(self, storage_path: str = "./user_profiles"):
        """
        Initialize profiler
        
        Args:
            storage_path: Path to store user profiles
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def extract_email(self, text: str) -> Optional[str]:
        """Extract email from text"""
        match = re.search(self.PATTERNS["email"], text)
        return match.group(0) if match else None

    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text"""
        match = re.search(self.PATTERNS["phone"], text)
        return match.group(0) if match else None

    def extract_age(self, text: str) -> Optional[str]:
        """Extract age from text"""
        match = re.search(self.PATTERNS["age"], text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 18 <= age <= 100:
                return str(age)
        return None

    def extract_investment_amount(self, text: str) -> Optional[float]:
        """Extract investment amount from text"""
        match = re.search(self.PATTERNS["investment_amount"], text)
        if match:
            amount_str = match.group(0)
            # Parse the amount
            amount_str = amount_str.replace("$", "").replace(",", "")
            if amount_str.lower().endswith("k"):
                return float(amount_str[:-1]) * 1000
            elif amount_str.lower().endswith("m"):
                return float(amount_str[:-1]) * 1000000
            else:
                try:
                    return float(amount_str)
                except ValueError:
                    pass
        return None

    def detect_investment_interests(self, text: str) -> List[str]:
        """Detect investment interests from text"""
        interests = []
        text_lower = text.lower()

        for interest_type, keywords in self.INTENT_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                interests.append(interest_type)

        return interests

    def detect_risk_tolerance(self, text: str) -> Optional[str]:
        """Detect risk tolerance from text"""
        text_lower = text.lower()

        for risk_level, keywords in self.RISK_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return risk_level

        return None

    def extract_name(self, text: str) -> Optional[str]:
        """
        Extract user name from text
        Looks for patterns like "I'm John" or "My name is Sarah"
        """
        patterns = [
            r"(?:my name is|i'?m|call me|name's?)\s+([A-Z][a-z]+)",
            r"(?:i'?m\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def extract_location(self, text: str) -> Optional[str]:
        """Extract location from text"""
        location_pattern = r"(?:in|from|live in|based in|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        match = re.search(location_pattern, text)
        return match.group(1) if match else None

    def extract_financial_goals(self, text: str) -> List[str]:
        """Extract financial goals from text"""
        goals = []

        goal_patterns = {
            "retirement": ["retire", "retirement", "retire early"],
            "wealth_building": ["build wealth", "get rich", "build net worth"],
            "passive_income": ["passive income", "generate income", "dividend"],
            "education_fund": ["education", "college fund", "school"],
            "home_purchase": ["buy home", "house", "mortgage"],
            "financial_independence": ["financial independence", "fire", "fi"]
        }

        text_lower = text.lower()

        for goal, keywords in goal_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                goals.append(goal)

        return goals

    def create_profile(self, user_id: str, query: str) -> UserProfile:
        """
        Create or update user profile from query
        
        Args:
            user_id: User identifier
            query: User query text
            
        Returns:
            UserProfile object
        """
        profile = UserProfile(user_id=user_id)

        # Extract all entities
        profile.name = self.extract_name(query)
        profile.email = self.extract_email(query)
        profile.phone = self.extract_phone(query)
        profile.age_group = self.extract_age(query)
        profile.investment_interests = self.detect_investment_interests(query)
        profile.risk_tolerance = self.detect_risk_tolerance(query)
        profile.location = self.extract_location(query)
        profile.financial_goals = self.extract_financial_goals(query)

        # Store query in conversation history
        profile.conversation_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "query": query
            }
        ]

        # Extract additional entities
        profile.extracted_entities = {
            "investment_amount": str(self.extract_investment_amount(query) or "Unknown")
        }

        return profile

    def update_profile(self, profile: UserProfile, query: str) -> UserProfile:
        """
        Update existing user profile with new query
        
        Args:
            profile: Existing profile
            query: New user query
            
        Returns:
            Updated profile
        """
        # Update or set name
        extracted_name = self.extract_name(query)
        if extracted_name and not profile.name:
            profile.name = extracted_name

        # Update or set email
        extracted_email = self.extract_email(query)
        if extracted_email and not profile.email:
            profile.email = extracted_email

        # Merge interests
        new_interests = self.detect_investment_interests(query)
        profile.investment_interests = list(
            set(profile.investment_interests + new_interests)
        )

        # Update risk tolerance if detected
        detected_risk = self.detect_risk_tolerance(query)
        if detected_risk and not profile.risk_tolerance:
            profile.risk_tolerance = detected_risk

        # Merge financial goals
        new_goals = self.extract_financial_goals(query)
        profile.financial_goals = list(
            set(profile.financial_goals + new_goals)
        )

        # Add to conversation history
        profile.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query
            }
        )

        profile.updated_at = datetime.now().isoformat()

        return profile

    def save_profile(self, profile: UserProfile) -> bool:
        """Save profile to storage"""
        try:
            profile_file = self.storage_path / f"{profile.user_id}.json"

            # Convert dataclass to dict for JSON serialization
            profile_dict = asdict(profile)

            with open(profile_file, 'w') as f:
                json.dump(profile_dict, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving profile: {e}")
            return False

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from storage"""
        try:
            profile_file = self.storage_path / f"{user_id}.json"

            if not profile_file.exists():
                return None

            with open(profile_file, 'r') as f:
                profile_dict = json.load(f)

            return UserProfile(**profile_dict)

        except Exception as e:
            print(f"Error loading profile: {e}")
            return None

    def get_profile_summary(self, profile: UserProfile) -> str:
        """Generate human-readable profile summary"""
        summary = []

        summary.append("=" * 50)
        summary.append("USER PROFILE SUMMARY")
        summary.append("=" * 50)

        if profile.name:
            summary.append(f"Name: {profile.name}")
        if profile.email:
            summary.append(f"Email: {profile.email}")
        if profile.age_group:
            summary.append(f"Age: {profile.age_group}")
        if profile.location:
            summary.append(f"Location: {profile.location}")

        if profile.investment_interests:
            summary.append(f"Investment Interests: {', '.join(profile.investment_interests)}")

        if profile.risk_tolerance:
            summary.append(f"Risk Tolerance: {profile.risk_tolerance}")

        if profile.financial_goals:
            summary.append(f"Financial Goals: {', '.join(profile.financial_goals)}")

        if profile.extracted_entities.get("investment_amount") != "Unknown":
            summary.append(f"Investment Amount: ${profile.extracted_entities.get('investment_amount')}")

        summary.append(f"Conversations: {len(profile.conversation_history)}")
        summary.append("=" * 50)

        return "\n".join(summary)
