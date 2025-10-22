"""
Main Application Orchestrator
Integrates KYC checking and AI customer support systems
"""

import sys
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

# Import KYC modules
from kyc_document_parser import KYCDocumentParser, DocumentType
from kyc_fraud_detector import KYCFraudDetector
from kyc_extractor import KYCExtractor
from kyc_storage import KYCStorage, KYCRecord

# Import Support modules
from support_nlp_engine import SupportNLPEngine
from support_user_profiler import SupportUserProfiler
from support_response_generator import SupportResponseGenerator


class FinancialAIAssistant:
    """Main AI assistant combining KYC and customer support"""

    def __init__(self, use_gemini: bool = False, gemini_key: Optional[str] = None):
        """
        Initialize the financial AI assistant
        
        Args:
            use_gemini: Whether to use Google Gemini API
            gemini_key: Google Gemini API key
        """
        print("🚀 Initializing Financial AI Assistant...")

        # Initialize KYC components
        self.kyc_parser = KYCDocumentParser()
        self.kyc_fraud_detector = KYCFraudDetector()
        self.kyc_extractor = KYCExtractor(use_ocr=True)
        self.kyc_storage = KYCStorage()

        # Initialize Support components
        self.nlp_engine = SupportNLPEngine(
            use_gemini=use_gemini,
            api_key=gemini_key
        )
        self.user_profiler = SupportUserProfiler()
        self.response_generator = SupportResponseGenerator(self.nlp_engine)

        print("✅ Assistant initialized successfully!\n")

    def run_kyc_verification(self, documents: List[Dict[str, str]]) -> Dict:
        """
        Run KYC verification on documents
        
        Args:
            documents: List of documents with 'text' and optional 'type'
            
        Returns:
            Verification result with status and details
        """
        print("=" * 70)
        print("🔐 KYC VERIFICATION PROCESS")
        print("=" * 70)

        # Parse documents
        print("\n1️⃣  Parsing documents...")
        parsed_docs = self.kyc_parser.parse_multiple_documents(documents)

        for i, doc in enumerate(parsed_docs, 1):
            print(f"   Document {i}: {doc.doc_type.value} (Confidence: {doc.confidence:.2%})")

        # Perform fraud detection
        print("\n2️⃣  Performing fraud detection...")
        fraud_result = self.kyc_fraud_detector.perform_fraud_check(parsed_docs)

        # Print fraud report
        print(self.kyc_fraud_detector.generate_fraud_report(fraud_result))

        # Store verification
        print("\n3️⃣  Storing verification record...")
        user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        verified_data = {}
        for doc in parsed_docs:
            verified_data.update(doc.extracted_fields)

        risk_level = "high" if fraud_result.is_fraud else "low"

        kyc_record = KYCRecord(
            user_id=user_id,
            verified_data=verified_data,
            documents=[doc.doc_type.value for doc in parsed_docs],
            verification_status="verified" if not fraud_result.is_fraud else "rejected",
            fraud_risk_level=risk_level,
            created_at=datetime.now().isoformat(),
            notes="; ".join(fraud_result.mismatches) if fraud_result.mismatches else "Verification successful"
        )

        self.kyc_storage.store_kyc_record(kyc_record)
        print(f"   ✅ Record stored with ID: {user_id}")

        return {
            "user_id": user_id,
            "status": "verified" if not fraud_result.is_fraud else "rejected",
            "fraud_detected": fraud_result.is_fraud,
            "fraud_confidence": fraud_result.confidence,
            "verified_data": verified_data,
            "warnings": fraud_result.warnings
        }

    def run_customer_support_session(self, user_id: str = None) -> None:
        """
        Run interactive customer support session
        
        Args:
            user_id: Optional existing user ID
        """
        print("\n" + "=" * 70)
        print("💬 AI-POWERED CUSTOMER SUPPORT SYSTEM")
        print("=" * 70)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'profile' to see your profile summary")
        print("Type 'report' to see your interaction report")
        print("=" * 70 + "\n")

        # Load or create user profile
        if user_id:
            profile = self.user_profiler.load_profile(user_id)
            if not profile:
                print(f"⚠️  User {user_id} not found. Creating new profile.\n")
                profile = None
        else:
            profile = None

        session_count = 0

        while True:
            try:
                # Get user input
                user_input = input("📝 You: ").strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Thank you for using Financial AI Assistant. Goodbye!")
                    break

                if user_input.lower() == 'profile':
                    if profile:
                        print("\n" + self.user_profiler.get_profile_summary(profile) + "\n")
                    else:
                        print("⚠️  No profile data yet. Start by asking a financial question!\n")
                    continue

                if user_input.lower() == 'report':
                    if profile:
                        print("\n" + self.response_generator.generate_summary_report(profile) + "\n")
                    else:
                        print("⚠️  No profile data yet. Start by asking a financial question!\n")
                    continue

                # Create or update profile
                if not profile:
                    user_session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    profile = self.user_profiler.create_profile(user_session_id, user_input)
                else:
                    profile = self.user_profiler.update_profile(profile, user_input)

                # Generate response
                contextual_response = self.response_generator.generate_contextual_response(
                    user_input,
                    profile
                )

                print(f"\n🤖 Assistant:\n{contextual_response['response']}\n")

                if contextual_response['follow_up_questions']:
                    print("💭 You might also want to know:")
                    print(contextual_response['follow_up_questions'])
                    print()

                session_count += 1

                # Save profile periodically
                if session_count % 3 == 0:
                    self.user_profiler.save_profile(profile)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                if profile:
                    self.user_profiler.save_profile(profile)
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

        # Final save
        if profile:
            self.user_profiler.save_profile(profile)
            print(f"📁 Profile saved with ID: {profile.user_id}")

    def demonstrate_kyc(self) -> None:
        """Demonstrate KYC verification with sample data"""
        print("\n" + "=" * 70)
        print("📋 KYC VERIFICATION DEMO")
        print("=" * 70 + "\n")

        # Sample documents
        sample_documents = [
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

        result = self.run_kyc_verification(sample_documents)

        print("\n✅ KYC Verification Result:")
        print(f"   Status: {result['status']}")
        print(f"   User ID: {result['user_id']}")
        print(f"   Fraud Detected: {result['fraud_detected']}")
        print(f"   Risk Level: {result['fraud_confidence']:.2%}\n")

    def demonstrate_support(self) -> None:
        """Demonstrate customer support with sample queries"""
        print("\n" + "=" * 70)
        print("💬 CUSTOMER SUPPORT DEMO")
        print("=" * 70 + "\n")

        sample_queries = [
            "Hi, I'm Sarah, 35 years old, interested in starting to invest. I have $50,000 and want to retire in 20 years. I'm conservative with risk.",
            "What's the best way to start investing in stocks?",
            "How do I handle a market crash?"
        ]

        user_id = f"demo_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        profile = None

        for i, query in enumerate(sample_queries, 1):
            print(f"Query {i}: {query}\n")

            # Create or update profile
            if not profile:
                profile = self.user_profiler.create_profile(user_id, query)
            else:
                profile = self.user_profiler.update_profile(profile, query)

            # Generate response
            response = self.response_generator.generate_contextual_response(query, profile)

            print(f"Response:\n{response['response']}\n")
            print("-" * 70 + "\n")

        # Save profile
        self.user_profiler.save_profile(profile)
        print(f"✅ Demo profile saved with ID: {user_id}\n")

    def show_statistics(self) -> None:
        """Show system statistics"""
        print("\n" + "=" * 70)
        print("📊 SYSTEM STATISTICS")
        print("=" * 70)

        # KYC Statistics
        kyc_stats = self.kyc_storage.get_statistics()
        print(f"\n🔐 KYC Verifications:")
        print(f"   Pending: {kyc_stats['pending']}")
        print(f"   Verified: {kyc_stats['verified']}")
        print(f"   Total: {kyc_stats['total']}")

        # OCR Capability
        ocr_capability = self.kyc_extractor.validate_ocr_capability()
        print(f"\n🖼️  OCR Capabilities:")
        print(f"   Pytesseract Available: {ocr_capability['pytesseract_available']}")
        print(f"   PIL Available: {ocr_capability['pil_available']}")
        print(f"   OCR Enabled: {ocr_capability['ocr_enabled']}")

        print("\n" + "=" * 70 + "\n")

    def interactive_menu(self) -> None:
        """Show interactive menu"""
        while True:
            print("=" * 70)
            print("🏦 FINANCIAL AI ASSISTANT - MAIN MENU")
            print("=" * 70)
            print("1. Run KYC Verification Demo")
            print("2. Run Customer Support Demo")
            print("3. Start Interactive Support Session")
            print("4. Show System Statistics")
            print("5. Exit")
            print("=" * 70)

            choice = input("Select an option (1-5): ").strip()

            if choice == "1":
                self.demonstrate_kyc()
            elif choice == "2":
                self.demonstrate_support()
            elif choice == "3":
                self.run_customer_support_session()
            elif choice == "4":
                self.show_statistics()
            elif choice == "5":
                print("\n👋 Thank you for using Financial AI Assistant!")
                break
            else:
                print("❌ Invalid option. Please try again.\n")


def main():
    """Main entry point"""
    print("\n" + "🏦" * 35)
    print("FINANCIAL AI ASSISTANT - KYC & CUSTOMER SUPPORT")
    print("🏦" * 35 + "\n")

    # Create assistant
    assistant = FinancialAIAssistant(use_gemini=True, gemini_key="YOUR_GEMINI_API_KEY")  # Set to True if using OpenAI

    # Run interactive menu
    assistant.interactive_menu()


if __name__ == "__main__":
    main()
