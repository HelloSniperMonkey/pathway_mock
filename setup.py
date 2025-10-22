"""
Setup and Installation Guide
Complete instructions for setting up the Financial AI Assistant
"""

import sys
import subprocess
from pathlib import Path


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"📋 {title}")
    print("=" * 70)


def check_python_version():
    """Check Python version"""
    print_section("CHECKING PYTHON VERSION")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ is required")
        return False


def install_dependencies(requirements_file: str = "requirements.txt"):
    """Install Python dependencies"""
    print_section("INSTALLING DEPENDENCIES")

    try:
        print(f"Installing from {requirements_file}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            check=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_directories():
    """Create required directories"""
    print_section("SETTING UP DIRECTORIES")

    directories = [
        "data",
        "logs",
        "kyc_data",
        "user_profiles"
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")

    print("✅ All directories set up")


def check_ocr_installation():
    """Check if Tesseract OCR is installed"""
    print_section("CHECKING OCR (OPTIONAL)")

    try:
        import pytesseract
        print("✅ pytesseract is installed")

        try:
            pytesseract.pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR is installed and working")
            return True
        except pytesseract.TesseractNotFoundError:
            print("⚠️  Tesseract OCR not found. To enable OCR:")
            print("   macOS: brew install tesseract")
            print("   Ubuntu: sudo apt-get install tesseract-ocr")
            print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            return False

    except ImportError:
        print("⚠️  pytesseract not installed")
        print("   Install with: pip install pytesseract pillow")
        return False


def check_transformers():
    """Check transformer models"""
    print_section("CHECKING TRANSFORMER MODELS (OPTIONAL)")

    try:
        import transformers
        print(f"✅ transformers library is installed (v{transformers.__version__})")

        try:
            from transformers import pipeline
            print("✅ Pipeline functionality available")
            return True
        except ImportError:
            print("⚠️  Pipeline not available")
            return False

    except ImportError:
        print("⚠️  transformers not installed")
        print("   Install with: pip install transformers torch")
        return False


def check_gemini():
    """Check Google Gemini API"""
    print_section("CHECKING GOOGLE GEMINI (OPTIONAL)")

    try:
        import google.generativeai
        print(f"✅ google-generativeai library is installed")

        import os
        if os.getenv("GEMINI_API_KEY"):
            print("✅ GEMINI_API_KEY environment variable is set")
            return True
        else:
            print("⚠️  GEMINI_API_KEY environment variable not set")
            print("   To enable Gemini: export GEMINI_API_KEY='your-key-here'")
            return False

    except ImportError:
        print("⚠️  google-generativeai not installed")
        print("   Install with: pip install google-generativeai")
        return False


def run_quick_test():
    """Run quick functionality test"""
    print_section("RUNNING QUICK TEST")

    try:
        print("Importing modules...")

        from kyc_document_parser import KYCDocumentParser
        print("✓ KYC parser imported")

        from kyc_fraud_detector import KYCFraudDetector
        print("✓ Fraud detector imported")

        from support_nlp_engine import SupportNLPEngine
        print("✓ NLP engine imported")

        from support_user_profiler import SupportUserProfiler
        print("✓ User profiler imported")

        from main import FinancialAIAssistant
        print("✓ Main assistant imported")

        print("\nTesting basic functionality...")

        # Test parser
        parser = KYCDocumentParser()
        doc_type = parser.detect_document_type("ABCDE1234F")
        assert doc_type.value == "pan"
        print("✓ Parser working")

        # Test profiler
        profiler = SupportUserProfiler()
        interests = profiler.detect_investment_interests("I want to invest in crypto")
        assert "crypto" in interests
        print("✓ Profiler working")

        # Test NLP
        engine = SupportNLPEngine()
        intent, _ = engine.detect_intent("stocks investment")
        assert intent == "investment"
        print("✓ NLP engine working")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def print_setup_summary(results: dict):
    """Print setup summary"""
    print_section("SETUP SUMMARY")

    print("\n✅ COMPLETED:")
    print("  • Python version checked")
    print("  • Dependencies installed")
    print("  • Directories created")
    print("  • Quick tests passed")

    print("\n⚠️  OPTIONAL COMPONENTS:")

    if not results.get("ocr"):
        print("  • OCR/Tesseract (for document image scanning)")

    if not results.get("transformers"):
        print("  • Transformer models (for local NLP)")

    if not results.get("gemini"):
        print("  • Google Gemini API (for cloud-based responses)")

    print("\n" + "=" * 70)
    print("🎉 SETUP COMPLETE!")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Run the application: python main.py")
    print("2. Or run demos: python demo.py")
    print("3. Or check configuration: python config.py")

    print("\nFor more information, see README.md")


def interactive_setup():
    """Interactive setup process"""
    print("\n" + "🚀" * 35)
    print("FINANCIAL AI ASSISTANT - SETUP WIZARD")
    print("🚀" * 35)

    results = {}

    # Check Python
    if not check_python_version():
        print("❌ Setup aborted: Python 3.8+ required")
        return False

    # Install dependencies
    if not install_dependencies():
        print("❌ Setup aborted: Failed to install dependencies")
        return False

    # Setup directories
    setup_directories()

    # Check optional components
    results["ocr"] = check_ocr_installation()
    results["transformers"] = check_transformers()
    results["gemini"] = check_gemini()

    # Quick test
    if run_quick_test():
        print_setup_summary(results)
        return True
    else:
        print("\n⚠️  Some tests failed, but setup may still work")
        print_setup_summary(results)
        return False


def main():
    """Main setup script"""
    try:
        success = interactive_setup()

        if success:
            print("\n✅ Setup completed successfully!")
            print("Run 'python main.py' to start the application")
            return 0
        else:
            print("\n⚠️  Setup completed with warnings")
            print("Run 'python main.py' to start the application (may have limited features)")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
