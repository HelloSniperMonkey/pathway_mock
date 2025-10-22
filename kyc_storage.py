"""
KYC Storage Module
Manages storage and retrieval of verified KYC data
"""

import json
import os
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class KYCRecord:
    """KYC verification record"""
    user_id: str
    verified_data: Dict[str, str]
    documents: List[str]
    verification_status: str  # "pending", "verified", "rejected"
    fraud_risk_level: str  # "low", "medium", "high"
    created_at: str
    verified_at: Optional[str] = None
    notes: str = ""


class KYCStorage:
    """Manages KYC data storage"""

    def __init__(self, storage_path: str = "./kyc_data"):
        """
        Initialize storage
        
        Args:
            storage_path: Path to store KYC records
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.verified_records_file = self.storage_path / "verified_records.json"
        self.pending_records_file = self.storage_path / "pending_records.json"

    def _load_records(self, file_path: Path) -> Dict[str, Dict]:
        """Load records from JSON file"""
        if not file_path.exists():
            return {}

        with open(file_path, 'r') as f:
            return json.load(f)

    def _save_records(self, file_path: Path, records: Dict[str, Dict]) -> None:
        """Save records to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(records, f, indent=2)

    def store_kyc_record(self, record: KYCRecord, is_verified: bool = False) -> str:
        """
        Store a KYC record
        
        Args:
            record: KYC record to store
            is_verified: Whether the record is verified
            
        Returns:
            User ID
        """
        file_path = self.verified_records_file if is_verified else self.pending_records_file
        records = self._load_records(file_path)

        records[record.user_id] = asdict(record)
        self._save_records(file_path, records)

        return record.user_id

    def retrieve_kyc_record(self, user_id: str) -> Optional[KYCRecord]:
        """
        Retrieve a KYC record by user ID
        
        Args:
            user_id: User ID
            
        Returns:
            KYCRecord if found, None otherwise
        """
        # Check verified records first
        verified_records = self._load_records(self.verified_records_file)
        if user_id in verified_records:
            data = verified_records[user_id]
            return KYCRecord(**data)

        # Check pending records
        pending_records = self._load_records(self.pending_records_file)
        if user_id in pending_records:
            data = pending_records[user_id]
            return KYCRecord(**data)

        return None

    def update_kyc_status(
        self,
        user_id: str,
        status: str,
        fraud_risk_level: str,
        notes: str = ""
    ) -> bool:
        """
        Update KYC verification status
        
        Args:
            user_id: User ID
            status: New status
            fraud_risk_level: Risk level
            notes: Additional notes
            
        Returns:
            True if updated successfully
        """
        # Try to find in pending records first
        pending_records = self._load_records(self.pending_records_file)

        if user_id in pending_records:
            record_data = pending_records[user_id]
            record = KYCRecord(**record_data)
            record.verification_status = status
            record.fraud_risk_level = fraud_risk_level
            record.notes = notes

            if status == "verified":
                record.verified_at = datetime.now().isoformat()
                # Move to verified records
                verified_records = self._load_records(self.verified_records_file)
                verified_records[user_id] = asdict(record)
                self._save_records(self.verified_records_file, verified_records)
                # Remove from pending
                del pending_records[user_id]
            else:
                # Keep in pending
                pending_records[user_id] = asdict(record)

            self._save_records(self.pending_records_file, pending_records)
            return True

        return False

    def list_pending_records(self) -> List[KYCRecord]:
        """List all pending KYC records"""
        pending_records = self._load_records(self.pending_records_file)
        return [KYCRecord(**data) for data in pending_records.values()]

    def list_verified_records(self) -> List[KYCRecord]:
        """List all verified KYC records"""
        verified_records = self._load_records(self.verified_records_file)
        return [KYCRecord(**data) for data in verified_records.values()]

    def get_statistics(self) -> Dict[str, int]:
        """Get KYC statistics"""
        pending = self._load_records(self.pending_records_file)
        verified = self._load_records(self.verified_records_file)

        return {
            "pending": len(pending),
            "verified": len(verified),
            "total": len(pending) + len(verified)
        }

    def export_record(self, user_id: str, export_format: str = "json") -> Optional[str]:
        """
        Export a KYC record
        
        Args:
            user_id: User ID
            export_format: Export format (json or csv)
            
        Returns:
            Exported data as string
        """
        record = self.retrieve_kyc_record(user_id)
        if not record:
            return None

        if export_format == "json":
            return json.dumps(asdict(record), indent=2)
        elif export_format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(asdict(record).keys())
            # Write data
            writer.writerow(asdict(record).values())

            return output.getvalue()

        return None
