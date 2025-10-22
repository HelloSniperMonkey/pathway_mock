"""
KYC Extractor Module
Extracts information from images using OCR (Optical Character Recognition)
Bonus feature: Supports both text and image documents
"""

import os
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class KYCExtractor:
    """Extracts text and data from KYC documents"""

    def __init__(self, use_ocr: bool = True):
        """
        Initialize extractor
        
        Args:
            use_ocr: Whether to use OCR for image processing
        """
        self.use_ocr = use_ocr and PYTESSERACT_AVAILABLE
        if use_ocr and not PYTESSERACT_AVAILABLE:
            print("Warning: pytesseract not available. Install with: pip install pytesseract pillow")

    def extract_from_image(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            (extracted_text, confidence_score)
        """
        if not self.use_ocr:
            return "", 0.0

        try:
            image = Image.open(image_path)

            # Preprocess image for better OCR accuracy
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Extract text
            extracted_text = pytesseract.image_to_string(image)

            # Calculate confidence (based on text length and clarity)
            confidence = min(1.0, len(extracted_text.strip()) / 500.0)

            return extracted_text, confidence

        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return "", 0.0

    def extract_from_text_file(self, file_path: str) -> Tuple[str, float]:
        """
        Extract text from text file
        
        Args:
            file_path: Path to text file
            
        Returns:
            (extracted_text, confidence_score)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Higher confidence for direct text files
            confidence = 1.0 if text.strip() else 0.0

            return text, confidence

        except Exception as e:
            print(f"Error reading text file: {e}")
            return "", 0.0

    def extract_from_document(self, document_path: str) -> Tuple[str, float]:
        """
        Extract text from any document (auto-detect type)
        
        Args:
            document_path: Path to document
            
        Returns:
            (extracted_text, confidence_score)
        """
        path = Path(document_path)

        if not path.exists():
            return "", 0.0

        # Get file extension
        extension = path.suffix.lower()

        # Image formats
        if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            return self.extract_from_image(document_path)

        # Text formats
        elif extension in ['.txt', '.csv', '.json']:
            return self.extract_from_text_file(document_path)

        else:
            print(f"Unsupported file format: {extension}")
            return "", 0.0

    def batch_extract(self, document_paths: list) -> Dict[str, Tuple[str, float]]:
        """
        Extract text from multiple documents
        
        Args:
            document_paths: List of document paths
            
        Returns:
            Dictionary mapping file paths to (text, confidence) tuples
        """
        results = {}

        for path in document_paths:
            text, confidence = self.extract_from_document(path)
            results[path] = (text, confidence)

        return results

    def validate_ocr_capability(self) -> Dict[str, bool]:
        """
        Validate OCR capabilities
        
        Returns:
            Dictionary with capability status
        """
        return {
            "pytesseract_available": PYTESSERACT_AVAILABLE,
            "pil_available": PYTESSERACT_AVAILABLE,
            "ocr_enabled": self.use_ocr
        }

    def preprocess_image(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Input image path
            output_path: Optional output path for preprocessed image
            
        Returns:
            True if successful
        """
        if not self.use_ocr:
            return False

        try:
            image = Image.open(image_path)

            # Convert to grayscale
            image = image.convert('L')

            # Increase contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)

            # Resize if too small
            width, height = image.size
            if width < 1000:
                scale = 1000 / width
                image = image.resize(
                    (int(width * scale), int(height * scale)),
                    Image.Resampling.LANCZOS
                )

            if output_path:
                image.save(output_path)

            return True

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return False

    def get_image_quality_score(self, image_path: str) -> float:
        """
        Estimate image quality for OCR
        
        Args:
            image_path: Path to image
            
        Returns:
            Quality score (0-1)
        """
        if not PYTESSERACT_AVAILABLE:
            return 0.0

        try:
            image = Image.open(image_path)

            # Check dimensions (larger is better for OCR)
            width, height = image.size
            size_score = min(1.0, (width * height) / 1000000.0)

            # Check if image is grayscale (better for OCR)
            mode_score = 1.0 if image.mode == 'L' else 0.7

            # Combined score
            quality = (size_score + mode_score) / 2

            return quality

        except Exception as e:
            print(f"Error calculating image quality: {e}")
            return 0.0


# Utility function to create extractor with optimal settings
def create_kyc_extractor(prefer_ocr: bool = True) -> KYCExtractor:
    """
    Factory function to create KYC extractor
    
    Args:
        prefer_ocr: Whether to use OCR if available
        
    Returns:
        KYCExtractor instance
    """
    return KYCExtractor(use_ocr=prefer_ocr)
