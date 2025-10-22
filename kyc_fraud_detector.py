"""
KYC Fraud Detection Module
Performs fraud detection by comparing data across documents
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from kyc_document_parser import ParsedDocument, DocumentType


@dataclass
class FraudCheckResult:
    """Result of fraud check"""
    is_fraud: bool
    confidence: float
    mismatches: List[str]
    warnings: List[str]


class KYCFraudDetector:
    """Detects fraud by cross-checking documents"""

    def __init__(self, name_match_threshold: float = 0.85):
        """
        Initialize fraud detector
        
        Args:
            name_match_threshold: Minimum similarity score for name matching
        """
        self.name_match_threshold = name_match_threshold
        self.critical_fields = ["name", "dob"]

    def string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity ratio"""
        s1 = str1.lower().strip()
        s2 = str2.lower().strip()
        return SequenceMatcher(None, s1, s2).ratio()

    def normalize_name(self, name: str) -> str:
        """Normalize name for comparison"""
        # Remove middle initials and extra spaces
        return " ".join(name.split()).lower()

    def check_name_consistency(self, names: List[str]) -> Tuple[bool, float, List[str]]:
        """
        Check if names are consistent across documents
        
        Returns:
            (is_consistent, confidence, mismatches)
        """
        if len(names) < 2:
            return True, 1.0, []

        normalized_names = [self.normalize_name(n) for n in names if n]
        if not normalized_names:
            return False, 0.0, ["No names found in documents"]

        # Compare first name with others
        first_name = normalized_names[0]
        mismatches = []

        for i, name in enumerate(normalized_names[1:], 1):
            similarity = self.string_similarity(first_name, name)
            if similarity < self.name_match_threshold:
                mismatches.append(
                    f"Name mismatch: '{names[0]}' vs '{names[i]}' "
                    f"(similarity: {similarity:.2%})"
                )

        is_consistent = len(mismatches) == 0
        confidence = 1.0 if is_consistent else max(0.0, 1.0 - len(mismatches) * 0.3)

        return is_consistent, confidence, mismatches

    def check_dob_consistency(self, dobs: List[str]) -> Tuple[bool, List[str]]:
        """Check if DOB is consistent across documents"""
        if len(dobs) < 2:
            return True, []

        # Normalize DOBs (remove separators)
        normalized_dobs = [dob.replace("/", "").replace("-", "") for dob in dobs if dob]
        if not normalized_dobs:
            return True, []

        mismatches = []
        first_dob = normalized_dobs[0]

        for i, dob in enumerate(normalized_dobs[1:], 1):
            if first_dob != dob:
                mismatches.append(
                    f"DOB mismatch: '{dobs[0]}' vs '{dobs[i]}'"
                )

        return len(mismatches) == 0, mismatches

    def check_data_completeness(self, parsed_docs: List[ParsedDocument]) -> Tuple[float, List[str]]:
        """
        Check completeness of extracted data
        
        Returns:
            (completeness_score, warnings)
        """
        warnings = []
        total_fields = 0
        extracted_fields = 0

        for doc in parsed_docs:
            if doc.doc_type != DocumentType.UNKNOWN:
                # Expected fields vary by document type
                if doc.doc_type == DocumentType.AADHAR:
                    expected = 5
                elif doc.doc_type == DocumentType.PAN:
                    expected = 4
                else:
                    expected = 4

                total_fields += expected
                extracted_fields += len(doc.extracted_fields)

                if doc.confidence < 0.6:
                    warnings.append(
                        f"Low extraction confidence for {doc.doc_type.value}: "
                        f"{doc.confidence:.2%}"
                    )

        completeness = (
            extracted_fields / total_fields
            if total_fields > 0
            else 0.0
        )

        if completeness < 0.7:
            warnings.append(f"Low data completeness: {completeness:.2%}")

        return completeness, warnings

    def detect_duplicates(self, parsed_docs: List[ParsedDocument]) -> List[str]:
        """Detect duplicate or similar documents"""
        warnings = []
        doc_hashes = {}

        for i, doc in enumerate(parsed_docs):
            # Create a simple hash of extracted fields
            doc_hash = tuple(sorted(doc.extracted_fields.items()))

            for doc_id, existing_hash in doc_hashes.items():
                if doc_hash == existing_hash:
                    warnings.append(
                        f"Potential duplicate: Document {doc_id} and {i}"
                    )

            doc_hashes[i] = doc_hash

        return warnings

    def perform_fraud_check(self, parsed_docs: List[ParsedDocument]) -> FraudCheckResult:
        """
        Perform comprehensive fraud check on multiple documents
        
        Args:
            parsed_docs: List of parsed documents
            
        Returns:
            FraudCheckResult with fraud detection results
        """
        if not parsed_docs:
            return FraudCheckResult(
                is_fraud=False,
                confidence=0.0,
                mismatches=[],
                warnings=["No documents provided"]
            )

        # Filter out unknown document types
        valid_docs = [d for d in parsed_docs if d.doc_type != DocumentType.UNKNOWN]

        if not valid_docs:
            return FraudCheckResult(
                is_fraud=False,
                confidence=0.0,
                mismatches=["No valid documents found"],
                warnings=[]
            )

        mismatches = []
        warnings = []

        # Check name consistency
        names = [doc.extracted_fields.get("name") for doc in valid_docs]
        names = [n for n in names if n]  # Filter out None values

        if len(names) > 1:
            name_consistent, name_confidence, name_mismatches = (
                self.check_name_consistency(names)
            )
            if not name_consistent:
                mismatches.extend(name_mismatches)

        # Check DOB consistency
        dobs = [doc.extracted_fields.get("dob") for doc in valid_docs]
        dobs = [d for d in dobs if d]  # Filter out None values

        if len(dobs) > 1:
            dob_consistent, dob_mismatches = self.check_dob_consistency(dobs)
            if not dob_consistent:
                mismatches.extend(dob_mismatches)

        # Check data completeness
        completeness, completeness_warnings = self.check_data_completeness(valid_docs)
        warnings.extend(completeness_warnings)

        # Check for duplicates
        duplicate_warnings = self.detect_duplicates(valid_docs)
        warnings.extend(duplicate_warnings)

        # Determine if fraud
        is_fraud = len(mismatches) > 0

        # Calculate confidence (0-1 scale)
        # Higher if no mismatches, lower if there are issues
        fraud_confidence = (
            min(1.0, len(mismatches) * 0.5) if mismatches else 0.0
        )

        return FraudCheckResult(
            is_fraud=is_fraud,
            confidence=fraud_confidence,
            mismatches=mismatches,
            warnings=warnings
        )

    def generate_fraud_report(self, check_result: FraudCheckResult) -> str:
        """Generate human-readable fraud check report"""
        report = []
        report.append("=" * 60)
        report.append("FRAUD DETECTION REPORT")
        report.append("=" * 60)

        fraud_status = "⚠️  FRAUD SUSPECTED" if check_result.is_fraud else "✅ NO FRAUD DETECTED"
        report.append(f"Status: {fraud_status}")
        report.append(f"Confidence: {check_result.confidence:.2%}")
        report.append("")

        if check_result.mismatches:
            report.append("MISMATCHES FOUND:")
            for mismatch in check_result.mismatches:
                report.append(f"  • {mismatch}")
            report.append("")

        if check_result.warnings:
            report.append("WARNINGS:")
            for warning in check_result.warnings:
                report.append(f"  • {warning}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)
