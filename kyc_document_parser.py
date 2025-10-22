"""
KYC Document Parser Module
Handles parsing of various KYC documents (Aadhar, PAN, Passport, etc.)
"""

import re
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class DocumentType(Enum):
    """Supported document types"""
    AADHAR = "aadhar"
    PAN = "pan"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    VOTER_ID = "voter_id"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocument:
    """Represents a parsed document"""
    doc_type: DocumentType
    raw_text: str
    extracted_fields: Dict[str, str]
    confidence: float


class KYCDocumentParser:
    """Parser for KYC documents"""

    # Regex patterns for Indian documents
    AADHAR_PATTERN = r"\b\d{4}\s\d{4}\s\d{4}\b"
    PAN_PATTERN = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"
    PASSPORT_PATTERN = r"[A-Z]{1}[0-9]{7}"

    def __init__(self):
        """Initialize the parser"""
        self.supported_doc_types = [
            DocumentType.AADHAR,
            DocumentType.PAN,
            DocumentType.PASSPORT,
            DocumentType.DRIVER_LICENSE,
            DocumentType.VOTER_ID
        ]

    def detect_document_type(self, text: str) -> DocumentType:
        """Detect the type of document from text"""
        text_upper = text.upper()

        if re.search(self.AADHAR_PATTERN, text):
            return DocumentType.AADHAR
        elif re.search(self.PAN_PATTERN, text):
            return DocumentType.PAN
        elif re.search(self.PASSPORT_PATTERN, text):
            return DocumentType.PASSPORT
        elif any(keyword in text_upper for keyword in ["DRIVER", "LICENSE", "DL"]):
            return DocumentType.DRIVER_LICENSE
        elif any(keyword in text_upper for keyword in ["VOTER", "ELECTORAL"]):
            return DocumentType.VOTER_ID
        else:
            return DocumentType.UNKNOWN

    def parse_aadhar(self, text: str) -> Dict[str, str]:
        """Parse Aadhar document"""
        fields = {}
        
        # Extract Aadhar number
        aadhar_match = re.search(self.AADHAR_PATTERN, text)
        if aadhar_match:
            fields["aadhar_number"] = aadhar_match.group(0).replace(" ", "")

        # Extract name (usually in uppercase at the start)
        name_pattern = r"(?:Name|नाम)\s*[:\-]?\s*([A-Za-z\s]+)"
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        if name_match:
            fields["name"] = name_match.group(1).strip()

        # Extract DOB
        dob_pattern = r"(?:DOB|Date of Birth|जन्म तिथि)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})"
        dob_match = re.search(dob_pattern, text, re.IGNORECASE)
        if dob_match:
            fields["dob"] = dob_match.group(1)

        # Extract gender
        gender_pattern = r"(?:Gender|लिंग)\s*[:\-]?\s*(M|F|Male|Female|पुरुष|महिला)"
        gender_match = re.search(gender_pattern, text, re.IGNORECASE)
        if gender_match:
            fields["gender"] = gender_match.group(1)

        # Extract address
        address_pattern = r"(?:Address|पता)\s*[:\-]?\s*([A-Za-z0-9\s,.-]+)"
        address_match = re.search(address_pattern, text, re.IGNORECASE)
        if address_match:
            fields["address"] = address_match.group(1).strip()

        return fields

    def parse_pan(self, text: str) -> Dict[str, str]:
        """Parse PAN document"""
        fields = {}

        # Extract PAN number
        pan_match = re.search(self.PAN_PATTERN, text)
        if pan_match:
            fields["pan_number"] = pan_match.group(0)

        # Extract name
        name_pattern = r"(?:Name|नाम)\s*[:\-]?\s*([A-Za-z\s]+)"
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        if name_match:
            fields["name"] = name_match.group(1).strip()

        # Extract father's name
        father_pattern = r"(?:Father|पिता)\s*[:\-]?\s*([A-Za-z\s]+)"
        father_match = re.search(father_pattern, text, re.IGNORECASE)
        if father_match:
            fields["father_name"] = father_match.group(1).strip()

        # Extract DOB
        dob_pattern = r"(?:DOB|Date of Birth|जन्म तिथि)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})"
        dob_match = re.search(dob_pattern, text, re.IGNORECASE)
        if dob_match:
            fields["dob"] = dob_match.group(1)

        return fields

    def parse_passport(self, text: str) -> Dict[str, str]:
        """Parse Passport document"""
        fields = {}

        # Extract passport number
        passport_match = re.search(self.PASSPORT_PATTERN, text)
        if passport_match:
            fields["passport_number"] = passport_match.group(0)

        # Extract name
        name_pattern = r"(?:Surname|Given Names|नाम)\s*[:\-]?\s*([A-Za-z\s]+)"
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        if name_match:
            fields["name"] = name_match.group(1).strip()

        # Extract nationality
        nationality_pattern = r"(?:Nationality|राष्ट्रीयता)\s*[:\-]?\s*([A-Za-z]+)"
        nationality_match = re.search(nationality_pattern, text, re.IGNORECASE)
        if nationality_match:
            fields["nationality"] = nationality_match.group(1).strip()

        # Extract DOB
        dob_pattern = r"(?:DOB|Date of Birth|जन्म तिथि)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})"
        dob_match = re.search(dob_pattern, text, re.IGNORECASE)
        if dob_match:
            fields["dob"] = dob_match.group(1)

        return fields

    def parse_document(self, text: str, doc_type: Optional[DocumentType] = None) -> ParsedDocument:
        """
        Parse a document and extract fields
        
        Args:
            text: Raw document text
            doc_type: Optional document type (will be auto-detected if not provided)
            
        Returns:
            ParsedDocument with extracted fields
        """
        if not doc_type:
            doc_type = self.detect_document_type(text)

        extracted_fields = {}
        confidence = 0.0

        if doc_type == DocumentType.AADHAR:
            extracted_fields = self.parse_aadhar(text)
            confidence = len(extracted_fields) / 5  # 5 possible fields
        elif doc_type == DocumentType.PAN:
            extracted_fields = self.parse_pan(text)
            confidence = len(extracted_fields) / 4  # 4 possible fields
        elif doc_type == DocumentType.PASSPORT:
            extracted_fields = self.parse_passport(text)
            confidence = len(extracted_fields) / 4  # 4 possible fields
        else:
            confidence = 0.0

        return ParsedDocument(
            doc_type=doc_type,
            raw_text=text,
            extracted_fields=extracted_fields,
            confidence=min(confidence, 1.0)
        )

    def parse_multiple_documents(self, documents: List[Dict[str, str]]) -> List[ParsedDocument]:
        """
        Parse multiple documents
        
        Args:
            documents: List of documents with 'text' and optional 'type' keys
            
        Returns:
            List of ParsedDocument objects
        """
        parsed_docs = []
        for doc in documents:
            doc_type = None
            if "type" in doc:
                try:
                    doc_type = DocumentType[doc["type"].upper()]
                except KeyError:
                    pass

            parsed = self.parse_document(doc["text"], doc_type)
            parsed_docs.append(parsed)

        return parsed_docs
