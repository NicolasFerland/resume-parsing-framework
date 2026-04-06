"""
Data models for the resume parsing framework.

This module defines the core data structures used throughout the system.
The ResumeData class serves as the immutable contract for extracted resume information.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass(frozen=True)
class ResumeData:
    """
    Immutable data structure representing extracted resume information.

    This frozen dataclass ensures that once resume data is extracted and structured,
    it cannot be accidentally modified. This prevents data corruption during processing
    and maintains data integrity throughout the pipeline.

    Fields:
        name: The candidate's full name (string)
        email: Primary email address (string)
        skills: List of technical and soft skills (list of strings)
    """

    name: str
    email: str
    skills: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the resume data object into a serializable dictionary.

        Returns a dictionary suitable for JSON serialization, maintaining
        the same field names and structure as the dataclass.
        """
        return asdict(self)
