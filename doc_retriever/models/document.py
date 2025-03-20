"""
Document model for storing document metadata and content.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class Document:
    """Document model representing a processed document."""
    id: str
    filename: str
    content: str
    chunks: List[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    file_size: int
    mime_type: str
    page_count: Optional[int] = None
    summary: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert document to dictionary format."""
        return {
            "id": self.id,
            "filename": self.filename,
            "content": self.content,
            "chunks": self.chunks,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Document':
        """Create document from dictionary format."""
        return cls(
            id=data["id"],
            filename=data["filename"],
            content=data["content"],
            chunks=data["chunks"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            file_size=data["file_size"],
            mime_type=data["mime_type"],
            page_count=data.get("page_count"),
            summary=data.get("summary")
        ) 