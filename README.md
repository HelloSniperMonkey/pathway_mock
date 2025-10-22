# 🏦 Financial AI Assistant - Complete AI System with Pathway Integration

A comprehensive AI-powered system implementing KYC verification, customer support, AI trading prediction, and Pathway streaming capabilities.

## 📋 Overview

This project implements **all four tasks** from the Pathway mock problem statement:

### 1. **Docker + Pathway Streaming** 🐳⚡
- Docker containerization for Linux environment
- Pathway library for real-time data processing
- Streaming stock price monitoring demo
- Sub-second latency data transformations

### 2. **Smart KYC Checker** 🔐
- Parse various identity documents (Aadhar, PAN, Passport, etc.)
- Extract relevant user information automatically
- Perform fraud detection by cross-checking documents
- **Bonus**: OCR support for scanning document images
- Store verified user data securely

### 3. **AI-Powered Customer Support** 💬
- Intelligent query processing using NLP
- Personalized responses based on user profile
- Extract and store user financial information from conversations
- **Bonus**: Build comprehensive user profiles over time
- **Bonus**: Google Gemini API integration

### 4. **AI Stock Price Prediction** 📈
- Machine learning models for price direction prediction
- Random Forest & Gradient Boosting classifiers
- 25+ technical indicators as features
- No future data leakage guarantee
- Professional visualizations and metrics

## 🏗️ Architecture

### Modular Design

```
main.py                          # Main orchestrator
pathway_demo.py                  # Pathway streaming demo (Task 1)
├── KYC SYSTEM (Task 2)
│   ├── kyc_document_parser.py      # Document parsing & type detection
│   ├── kyc_extractor.py            # OCR & text extraction (BONUS)
│   ├── kyc_fraud_detector.py       # Fraud detection & cross-validation
│   └── kyc_storage.py              # Data persistence & retrieval
├── SUPPORT SYSTEM (Task 3)
│   ├── support_nlp_engine.py       # Gemini NLP processing
│   ├── support_user_profiler.py    # User profile building (BONUS)
│   └── support_response_generator.py # Contextual response generation
└── TRADING AI SYSTEM (Task 4)
    ├── ai_price_predictor.py       # ML prediction engine
    ├── integrated_ai_trading.py    # Complete pipeline
    └── trading_integration.py      # System integration
```

## 🚀 Features

### Task 1: Pathway Streaming Features
✅ **Real-time Data Processing**
- Pathway library integration
- CSV streaming with transformations
- Data aggregations and filtering
- Sub-second latency processing

✅ **Docker Containerization**
- Python 3.10-slim base image
- All dependencies pre-installed
- Ready-to-run Pathway demo
- Production-ready configuration

### Task 2: KYC Verification Features
✅ **Multi-document Support**
- Aadhar, PAN, Passport, Driver's License, Voter ID
- Auto-detection of document type
- Regex-based field extraction

✅ **Fraud Detection**
- Name consistency checking
- Date of Birth validation
- Duplicate detection
- Data completeness analysis
- Risk level assessment

✅ **OCR Capability (BONUS)**
- Image-based document scanning
- Automatic image preprocessing
- Quality score evaluation
- Batch processing support

✅ **Data Storage & Management**
- JSON-based persistence
- Verification status tracking
- Audit trails
- Export capabilities (JSON, CSV)

### Customer Support Features
✅ **Intelligent Query Processing**
- Intent detection (investment, trading, crypto, etc.)
- Frequently Asked Questions (FAQ) matching
- Multi-LLM support (Transformers, OpenAI)
- Context awareness

✅ **User Profile Building (BONUS)**
- Automatic entity extraction (name, email, age, location)
- Investment interest detection
- Risk tolerance assessment
- Financial goal identification
- Conversation history tracking

✅ **Personalized Responses**
- Profile-based customization
- Expertise level adaptation
- Goal-specific recommendations
- Follow-up question generation
- Comprehensive interaction reports

## 📦 Installation

### Prerequisites
- Docker Desktop (for Task 1)
- Python 3.8+ (for local development)

### Quick Start with Docker (Recommended)

**Step 1: Build Docker image**
```bash
docker build -t financial-ai-assistant .
```

**Step 2: Run Pathway demo (Task 1)**
```bash
docker run -it financial-ai-assistant
```

This will execute `pathway_demo.py` showing Pathway streaming capabilities.

**Step 3: Run main Financial AI system**
```bash
docker run -it financial-ai-assistant python main.py
```

**Step 4: Run AI Trading prediction**
```bash
docker run -it financial-ai-assistant python integrated_ai_trading.py
```

### Local Installation (Without Docker)

```

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone/Download the project**
```bash
cd pathway_mock
```

2. **Create virtual environment (optional but recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Optional: Install Tesseract OCR** (for better image processing)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 🎯 Usage

### Quick Start

```bash
python main.py
```

This launches the interactive menu with options to:
1. Run KYC Verification Demo
2. Run Customer Support Demo
3. Start Interactive Support Session
4. View System Statistics

### Example: KYC Verification

```python
from main import FinancialAIAssistant

# Initialize assistant
assistant = FinancialAIAssistant()

# Prepare documents
documents = [
    {
        "type": "aadhar",
        "text": "Aadhar Number: 1234 5678 9012\nName: John Doe\nDOB: 15/03/1990"
    },
    {
        "type": "pan",
        "text": "PAN: ABCDE1234F\nName: John Doe\nDOB: 15/03/1990"
    }
]

# Run verification
result = assistant.run_kyc_verification(documents)
```

### Example: Customer Support Session

```python
from main import FinancialAIAssistant

# Initialize assistant
assistant = FinancialAIAssistant()

# Start interactive session
assistant.run_customer_support_session()
```

## 📊 Module Documentation

### `kyc_document_parser.py`
**Purpose**: Parse and extract data from KYC documents

**Key Classes**:
- `KYCDocumentParser`: Main parser with document type detection
- `ParsedDocument`: Data class for parsed document results

**Key Methods**:
```python
# Auto-detect document type
doc_type = parser.detect_document_type(text)

# Parse specific document
parsed = parser.parse_document(text, DocumentType.AADHAR)

# Parse multiple documents
parsed_docs = parser.parse_multiple_documents(documents)
```

### `kyc_extractor.py`
**Purpose**: Extract text from documents using OCR (BONUS feature)

**Key Classes**:
- `KYCExtractor`: Handles text extraction from images and files

**Key Methods**:
```python
# Extract from image using OCR
text, confidence = extractor.extract_from_image(image_path)

# Extract from any document
text, confidence = extractor.extract_from_document(doc_path)

# Batch processing
results = extractor.batch_extract(document_paths)

# Preprocess image for better OCR
extractor.preprocess_image(input_path, output_path)

# Quality assessment
quality_score = extractor.get_image_quality_score(image_path)
```

### `kyc_fraud_detector.py`
**Purpose**: Detect fraudulent documents through cross-validation

**Key Classes**:
- `KYCFraudDetector`: Performs comprehensive fraud checks
- `FraudCheckResult`: Results data class

**Key Methods**:
```python
# Check name consistency
is_consistent, confidence, mismatches = detector.check_name_consistency(names)

# Check DOB consistency
is_consistent, mismatches = detector.check_dob_consistency(dobs)

# Full fraud check
result = detector.perform_fraud_check(parsed_docs)

# Generate report
report = detector.generate_fraud_report(result)
```

### `kyc_storage.py`
**Purpose**: Persist and retrieve KYC verification records

**Key Classes**:
- `KYCStorage`: File-based storage manager
- `KYCRecord`: Verification record data class

**Key Methods**:
```python
# Store record
user_id = storage.store_kyc_record(record)

# Retrieve record
record = storage.retrieve_kyc_record(user_id)

# Update status
storage.update_kyc_status(user_id, "verified", "low_risk")

# Get statistics
stats = storage.get_statistics()

# Export
exported = storage.export_record(user_id, format="json")
```

### `support_nlp_engine.py`
**Purpose**: Process queries with NLP and generate responses

**Key Classes**:
- `SupportNLPEngine`: NLP processing engine

**Key Methods**:
```python
# Detect query intent
intent, confidence = engine.detect_intent(query)

# Process query
result = engine.process_query(query, context="")

# Validate response
validations = engine.validate_response(query, response)
```

### `support_user_profiler.py`
**Purpose**: Build user profiles from conversations (BONUS)

**Key Classes**:
- `SupportUserProfiler`: Profile building and management
- `UserProfile`: User data class

**Key Methods**:
```python
# Create profile
profile = profiler.create_profile(user_id, query)

# Update profile
profile = profiler.update_profile(profile, query)

# Extract entities
name = profiler.extract_name(text)
interests = profiler.detect_investment_interests(text)
goals = profiler.extract_financial_goals(text)

# Persistence
profiler.save_profile(profile)
profile = profiler.load_profile(user_id)

# Get summary
summary = profiler.get_profile_summary(profile)
```

### `support_response_generator.py`
**Purpose**: Generate personalized contextual responses (BONUS)

**Key Classes**:
- `SupportResponseGenerator`: Personalized response generation

**Key Methods**:
```python
# Generate contextual response
response = generator.generate_contextual_response(query, profile)

# Generate follow-up questions
follow_ups = generator.generate_follow_up_questions(profile, query)

# Generate interaction report
report = generator.generate_summary_report(profile)
```

## 📁 Data Structure

### KYC Verification Record
```json
{
  "user_id": "user_20250101_120000",
  "verified_data": {
    "name": "John Doe",
    "aadhar_number": "1234567890123",
    "dob": "15/03/1990",
    "address": "123 Main St"
  },
  "documents": ["aadhar", "pan"],
  "verification_status": "verified",
  "fraud_risk_level": "low",
  "created_at": "2025-01-01T12:00:00",
  "verified_at": "2025-01-01T12:05:00",
  "notes": "Verification successful"
}
```

### User Profile Record
```json
{
  "user_id": "user_20250101_120000",
  "name": "Sarah",
  "email": "sarah@example.com",
  "age_group": "35",
  "investment_interests": ["stocks", "crypto"],
  "risk_tolerance": "conservative",
  "financial_goals": ["retirement", "wealth_building"],
  "conversation_history": [
    {
      "timestamp": "2025-01-01T12:00:00",
      "query": "How do I start investing?"
    }
  ]
}
```

## 🎁 BONUS Features Implemented

### 1. **OCR Integration**
- Document image scanning and text extraction
- Image preprocessing for better OCR accuracy
- Quality score evaluation
- Batch processing capabilities

### 2. **User Profile Building**
- Automatic entity extraction from conversations
- Investment interest detection
- Risk tolerance assessment
- Financial goal identification
- Comprehensive profile persistence
- Interaction history tracking

### 3. **Personalized Responses**
- Profile-based response customization
- Expertise level adaptation
- Goal-specific recommendations
- Follow-up question generation
- Comprehensive interaction reports
- User expertise progression tracking

### 4. **Multi-LLM Support**
- Transformer models support (Hugging Face)
- OpenAI API integration ready
- Fallback strategies for degraded functionality
- FAQ-based response generation

## 🔒 Security Considerations

1. **Data Privacy**: Store sensitive information securely
2. **Input Validation**: Validate all extracted data
3. **Fraud Detection**: Cross-validate documents
4. **Access Control**: Implement user authentication
5. **Encryption**: Encrypt stored sensitive data

## 📈 Performance Optimization

1. **Batch Processing**: Process multiple documents efficiently
2. **Caching**: Cache FAQ responses
3. **Profile Caching**: Load profiles only when needed
4. **Lazy Loading**: Initialize heavy components on demand

## 🧪 Testing

Run the application and test with the demo features:

```bash
python main.py
# Select option 1 for KYC demo
# Select option 2 for Support demo
# Select option 3 for interactive session
```

## 📝 Example Interactions

### KYC Verification
```
1️⃣  Parsing documents...
   Document 1: aadhar (Confidence: 100.00%)
   Document 2: pan (Confidence: 80.00%)

2️⃣  Performing fraud detection...
✅ NO FRAUD DETECTED
Confidence: 0.00%

3️⃣  Storing verification record...
   ✅ Record stored with ID: user_20250101_120000
```

### Customer Support
```
📝 You: Hi, I'm Sarah, 35 years old, interested in crypto

🤖 Assistant:
Hello Sarah! I see you're interested in crypto.

📌 Based on your profile considerations:
• Start with small crypto investments
• Understand the technology first
• Use secure wallets

💭 You might also want to know:
• How experienced are you with cryptocurrency?
• What is your investment timeline?
```

## 🐛 Troubleshooting

### OCR Not Working
- Ensure Tesseract is installed: `brew install tesseract`
- Check pytesseract is installed: `pip install pytesseract`
- Verify image quality and format

### OpenAI API Not Responding
- Check API key validity
- Ensure internet connection
- Fall back to transformer models

### Profile Not Saving
- Ensure `./user_profiles` directory is writable
- Check disk space
- Verify file permissions

## 🤝 Contributing

Feel free to extend this project with:
- Additional document types
- More fraud detection rules
- Enhanced NLP models
- Database integration
- API endpoints
- Web UI

## 📄 License

This project is provided as-is for educational and commercial use.

## 🎓 Learning Resources

- Document Parsing: Pattern matching, regex
- Fraud Detection: Cross-validation, similarity matching
- OCR: Tesseract, image preprocessing
- NLP: Intent detection, entity extraction
- Profile Building: User modeling, conversation analysis
- AI Integration: LLM APIs, fallback strategies

---

**Last Updated**: January 2025
**Version**: 1.0.0
**Author**: AI Assistant Team
