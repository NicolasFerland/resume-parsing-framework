import logging
import docx2txt
from core.interfaces import FileParser

logger = logging.getLogger(__name__)

class WordParser(FileParser):
    """
    Parser for Microsoft Word documents (.docx format).

    Uses the docx2txt library to extract plain text from Word documents.
    Handles various Word document formats and provides error handling for
    corrupted or unreadable files.
    """
    def extract_text(self, file_path: str) -> str:
        """Extracts text content from a DOCX file.

        Args:
            file_path (str): The path to the .docx file.

        Returns:
            str: The extracted plain text.
            
        Raises:
            RuntimeError: If the DOCX file is corrupted or unreadable.
        """
        try:
            text = docx2txt.process(file_path)
            if text is None:
                logger.warning(f"DOCX file appears to be empty or unreadable: {file_path}")
                return ""
            
            logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}")
            raise RuntimeError(f"Could not read DOCX: {e}")