"""
Demo and Testing Script
Demonstrates all features of the Financial AI Assistant
"""

from main import FinancialAIAssistant
from config import print_config, validate_config
from datetime import datetime


def print_header(title: str) -> None:
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"🎬 {title}")
    print("=" * 70 + "\n")


def demo_kyc_basic() -> None:
    """Demo: Basic KYC verification"""
    print_header("KYC VERIFICATION - BASIC DEMO")

    assistant = FinancialAIAssistant()

    # Sample documents
    documents = [
        {
            "type": "aadhar",
            "text": """
            Aadhar Number: 1234 5678 9012
            Name: John Smith
            Date of Birth: 15/03/1990
            Gender: Male
            Address: 123 Main Street, Springfield, IL 62701
            """
        },
        {
            "type": "pan",
            "text": """
            PAN Number: ABCDE1234F
            Name: John Smith
            Father's Name: Robert Smith
            Date of Birth: 15/03/1990
            """
        }
    ]

    print("📄 Documents to verify:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['type'].upper()}")

    result = assistant.run_kyc_verification(documents)

    print("\nResults:")
    print(f"  User ID: {result['user_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Fraud Detected: {'Yes' if result['fraud_detected'] else 'No'}")
    print(f"  Risk Level: {result['fraud_confidence']:.2%}")


def demo_kyc_fraud() -> None:
    """Demo: KYC verification with fraud detection"""
    print_header("KYC VERIFICATION - FRAUD DETECTION DEMO")

    assistant = FinancialAIAssistant()

    # Mismatched documents
    documents = [
        {
            "type": "aadhar",
            "text": """
            Aadhar Number: 1234 5678 9012
            Name: John Smith
            Date of Birth: 15/03/1990
            """
        },
        {
            "type": "pan",
            "text": """
            PAN Number: ABCDE1234F
            Name: Jane Smith
            Date of Birth: 20/05/1985
            """
        }
    ]

    print("📄 Documents with potential issues:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['type'].upper()}")

    result = assistant.run_kyc_verification(documents)

    if result['fraud_detected']:
        print("\nFRAUD INDICATORS DETECTED!")
        print("Details:")
        for warning in result['warnings']:
            print(f"  • {warning}")


def demo_customer_support() -> None:
    """Demo: Customer support interaction"""
    print_header("CUSTOMER SUPPORT - INTERACTIVE DEMO")

    assistant = FinancialAIAssistant()

    sample_interactions = [
        {
            "query": "Hi, I'm Sarah, 35 years old, and I want to start investing. I have $50,000 saved up.",
            "label": "Profile Creation"
        },
        {
            "query": "I'm quite conservative. What's the safest way to invest my money?",
            "label": "Risk Tolerance"
        },
        {
            "query": "I want to retire in 20 years. How should I plan?",
            "label": "Financial Goal"
        },
        {
            "query": "Should I invest in cryptocurrencies?",
            "label": "Crypto Interest"
        }
    ]

    assistant.demonstrate_support()


def demo_statistics() -> None:
    """Demo: System statistics"""
    print_header("SYSTEM STATISTICS")

    assistant = FinancialAIAssistant()
    assistant.show_statistics()


def demo_full_workflow() -> None:
    """Demo: Complete workflow"""
    print_header("COMPLETE WORKFLOW DEMO")

    print("Step 1: Verify configuration")
    print_config()

    if not validate_config():
        print("Warning: Configuration validation failed. Continuing anyway...\n")

    print("\nStep 2: Initialize assistant")
    assistant = FinancialAIAssistant()
    print("Assistant initialized\n")

    print("\nStep 3: Run KYC demo")
    input("Press Enter to continue...")
    demo_kyc_basic()

    print("\n\nStep 4: Run fraud detection demo")
    input("Press Enter to continue...")
    demo_kyc_fraud()

    print("\n\nStep 5: View statistics")
    input("Press Enter to continue...")
    demo_statistics()

    print("\n\nStep 6: Run customer support demo")
    input("Press Enter to continue...")
    demo_customer_support()

    print_header("WORKFLOW COMPLETE")


def run_quick_test() -> None:
    """Run quick test of all systems"""
    print_header("QUICK SYSTEM TEST")

    try:
        # Test 1: Initialize
        print("Test 1: Initializing assistant...")
        assistant = FinancialAIAssistant()
        print("Pass\n")

        # Test 2: KYC
        print("Test 2: Testing KYC module...")
        from kyc_document_parser import KYCDocumentParser
        parser = KYCDocumentParser()
        doc_type = parser.detect_document_type("PAN: ABCDE1234F")
        assert doc_type.value == "pan"
        print("Pass\n")

        # Test 3: User Profiler
        print("Test 3: Testing user profiler...")
        from support_user_profiler import SupportUserProfiler
        profiler = SupportUserProfiler()
        name = profiler.extract_name("My name is John Smith")
        assert name == "John Smith"
        print("Pass\n")

        # Test 4: NLP Engine
        print("Test 4: Testing NLP engine...")
        from support_nlp_engine import SupportNLPEngine
        engine = SupportNLPEngine()
        intent, conf = engine.detect_intent("I want to invest in stocks")
        assert intent == "investment"
        print("Pass\n")

        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70 + "\n")

    except AssertionError as e:
        print(f"Test failed: {e}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def main():
    """Main demo launcher"""
    print("\n" + "🎬" * 35)
    print("FINANCIAL AI ASSISTANT - DEMO SUITE")
    print("🎬" * 35 + "\n")

    menu_options = {
        "1": ("Quick System Test", run_quick_test),
        "2": ("Basic KYC Verification", demo_kyc_basic),
        "3": ("Fraud Detection Demo", demo_kyc_fraud),
        "4": ("Customer Support Demo", demo_customer_support),
        "5": ("System Statistics", demo_statistics),
        "6": ("Full Workflow Demo", demo_full_workflow),
        "7": ("Interactive Menu", lambda: FinancialAIAssistant().interactive_menu()),
    }

    while True:
        print("=" * 70)
        print("SELECT A DEMO")
        print("=" * 70)

        for key, (name, _) in menu_options.items():
            print(f"{key}. {name}")

        print("8. Exit")
        print("=" * 70)

        choice = input("Select option (1-8): ").strip()

        if choice == "8":
            print("\nThanks for using the demo suite!")
            break
        elif choice in menu_options:
            try:
                _, func = menu_options[choice]
                func()
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user\n")
            except Exception as e:
                print(f"\nError: {e}\n")
        else:
            print("Invalid option. Please try again.\n")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
