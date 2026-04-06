"""
Core orchestration components for the resume parsing framework.

This module contains the main coordination classes that manage the parsing
and extraction workflow. The ResumeExtractor follows a consistent error
handling pattern where missing extractors or extraction failures result
in safe default values rather than system crashes.
"""

import os
import logging
from typing import Dict
from core.interfaces import FileParser, FieldExtractor
from core.models import ResumeData

logger = logging.getLogger(__name__)


class ResumeExtractor:
    """Orchestrates the extraction of multiple fields from raw text."""

    def __init__(self, extractors: Dict[str, FieldExtractor]):
        """
        Initializes the coordinator with specific field strategies.

        Args:
            extractors (Dict[str, FieldExtractor]): Mapping of field names to
                extractors.
        """
        if not extractors:
            raise ValueError("At least one extractor must be provided.")
        self.extractors = extractors

    def orchestrate(self, text: str) -> ResumeData:
        """
        Triggers each extractor and aggregates results into a ResumeData object.

        This method follows a consistent error handling pattern for all extractors:
        - Missing extractors (KeyError) result in safe default values
        - Extraction failures (Exception) are logged and default values are used
        - The system continues processing even if individual fields fail

        This graceful degradation ensures the system remains functional even with
        partial configuration or temporary extraction issues.

        Args:
            text (str): The raw resume text.

        Returns:
            ResumeData: The final structured resume object.
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid or empty text provided for extraction.")
            return ResumeData(name="Unknown", email="Not Found", skills=[])

        # Extract name with error handling
        try:
            name = self.extractors["name"].extract(text)
        except KeyError:
            logger.error("Name extractor not configured.")
            name = "Unknown"
        except Exception as e:
            logger.error(f"Error during name extraction: {e}")
            name = "Unknown"

        # Extract email with same error handling pattern
        try:
            email = self.extractors["email"].extract(text)
        except KeyError:
            logger.error("Email extractor not configured.")
            email = "Not Found"
        except Exception as e:
            logger.error(f"Error during email extraction: {e}")
            email = "Not Found"

        # Extract skills with same error handling pattern
        try:
            skills = self.extractors["skills"].extract(text)
        except KeyError:
            logger.error("Skills extractor not configured.")
            skills = []
        except Exception as e:
            logger.error(f"Error during skills extraction: {e}")
            skills = []

        logger.info("Extraction orchestration completed.")
        return ResumeData(name=name, email=email, skills=skills)


class ResumeParserFramework:
    """Facade for the resume parsing system."""

    def __init__(
        self, parsers: Dict[str, FileParser], extractor_coordinator: ResumeExtractor
    ):
        """
        Initializes the framework with supported parsers and an extractor.

        Args:
            parsers (Dict[str, FileParser]): Mapping of extensions to parser objects.
            extractor_coordinator (ResumeExtractor): The engine that handles field
                extraction.
        """
        self._parsers = parsers
        self._coordinator = extractor_coordinator

    def parse_resume(self, file_path: str) -> ResumeData:
        """
        Main entry point for parsing a single resume file.

        Args:
            file_path (str): Path to the .pdf or .docx file.

        Returns:
            ResumeData: Structured data extracted from the file.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path provided.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file does not exist: {file_path}")

        ext = os.path.splitext(file_path.lower())[1]
        if not ext:
            raise ValueError(f"No file extension found for: {file_path}")

        parser = self._parsers.get(ext)

        if not parser:
            raise ValueError(f"No parser registered for extension: {ext}")

        try:
            raw_text = parser.extract_text(file_path)
            if not raw_text:
                logger.warning(f"No text extracted from {file_path}")
            return self._coordinator.orchestrate(raw_text)
        except Exception as e:
            logger.error(f"Failed to parse resume {file_path}: {e}")
            raise
