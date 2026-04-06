from abc import ABC, abstractmethod
from typing import Any


class FileParser(ABC):
    """Abstract base class for all file format parsers."""

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Converts a file into raw string text.

        Args:
            file_path (str): The physical path to the resume file.

        Returns:
            str: The extracted raw text.
        """
        ...


class FieldExtractor(ABC):
    """Abstract base class for individual field extraction strategies."""

    @abstractmethod
    def extract(self, text: str) -> Any:
        """
        Extracts a specific data point from the provided text.

        Args:
            text (str): The raw text of the resume.

        Returns:
            Any: The structured data (string, list, etc.) for the field.
        """
        ...
