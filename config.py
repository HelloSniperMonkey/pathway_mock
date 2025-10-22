"""
Configuration Module
Centralized configuration for the Financial AI Assistant
"""

import os
from pathlib import Path
from typing import Optional

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
KYC_DATA_DIR = PROJECT_ROOT / "kyc_data"
PROFILES_DIR = PROJECT_ROOT / "user_profiles"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, KYC_DATA_DIR, PROFILES_DIR]:
    directory.mkdir(exist_ok=True)

# API Configuration
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
USE_GEMINI: bool = os.getenv("USE_GEMINI", "false").lower() == "true"

# Model Configuration
NLP_MODEL: str = os.getenv("NLP_MODEL", "google/flan-t5-small")
USE_TRANSFORMERS: bool = True  # Always try transformers first

# OCR Configuration
USE_OCR: bool = True
OCR_LANG: str = "eng"  # English by default
PYTESSERACT_PATH: Optional[str] = os.getenv("PYTESSERACT_PATH", None)

# KYC Configuration
KYC_STORAGE_PATH: str = str(KYC_DATA_DIR)
KYC_NAME_MATCH_THRESHOLD: float = 0.85
KYC_FRAUD_CONFIDENCE_THRESHOLD: float = 0.5

# Support Configuration
SUPPORT_STORAGE_PATH: str = str(PROFILES_DIR)
SUPPORT_SESSION_TIMEOUT: int = 300  # 5 minutes in seconds
SUPPORT_MAX_FOLLOW_UP_QUESTIONS: int = 3

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = str(LOGS_DIR / "financial_assistant.log")

# FAQ Configuration
FAQ_CACHE_ENABLED: bool = True
FAQ_CACHE_TTL: int = 3600  # 1 hour

# Security Configuration
ENCRYPTION_ENABLED: bool = False  # Can be enabled for production
DATA_RETENTION_DAYS: int = 365
PII_REDACTION_ENABLED: bool = True  # Redact personally identifiable information in logs

# Feature Flags
FEATURES = {
    "kyc_verification": True,
    "customer_support": True,
    "ocr_processing": True,
    "user_profiling": True,
    "fraud_detection": True,
    "llm_responses": True,
}

# Supported Document Types
SUPPORTED_DOCUMENT_TYPES = [
    "aadhar",
    "pan",
    "passport",
    "driver_license",
    "voter_id"
]

# Investment Categories
INVESTMENT_CATEGORIES = [
    "stocks",
    "crypto",
    "real_estate",
    "bonds",
    "mutual_funds",
    "trading",
    "retirement",
    "savings"
]

# Risk Tolerance Levels
RISK_TOLERANCE_LEVELS = ["conservative", "moderate", "aggressive"]

# Financial Goals
FINANCIAL_GOALS = [
    "retirement",
    "wealth_building",
    "passive_income",
    "education_fund",
    "home_purchase",
    "financial_independence"
]


def get_config() -> dict:
    """Get all configuration as dictionary"""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "logs_dir": str(LOGS_DIR),
        "kyc_data_dir": str(KYC_DATA_DIR),
        "profiles_dir": str(PROFILES_DIR),
        "gemini_api_key": "SET" if GEMINI_API_KEY else "NOT SET",
        "use_gemini": USE_GEMINI,
        "nlp_model": NLP_MODEL,
        "use_transformers": USE_TRANSFORMERS,
        "use_ocr": USE_OCR,
        "ocr_lang": OCR_LANG,
        "kyc_storage_path": KYC_STORAGE_PATH,
        "support_storage_path": SUPPORT_STORAGE_PATH,
        "log_level": LOG_LEVEL,
        "features": FEATURES,
    }


def validate_config() -> bool:
    """Validate configuration"""
    errors = []

    # Check required directories
    for directory in [DATA_DIR, LOGS_DIR, KYC_DATA_DIR, PROFILES_DIR]:
        if not directory.exists():
            errors.append(f"Directory does not exist: {directory}")

    # Check Gemini configuration
    if USE_GEMINI and not GEMINI_API_KEY:
        errors.append("USE_GEMINI is True but GEMINI_API_KEY is not set")

    if errors:
        print("⚠️  Configuration Errors:")
        for error in errors:
            print(f"  • {error}")
        return False

    return True


def print_config() -> None:
    """Print current configuration"""
    config = get_config()

    print("\n" + "=" * 70)
    print("⚙️  CONFIGURATION")
    print("=" * 70)

    for key, value in config.items():
        print(f"{key:.<40} {value}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_config()
    if validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors")
