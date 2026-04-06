import re
import logging
from core.interfaces import FieldExtractor

# Initialize logger for this module
logger = logging.getLogger(__name__)


class EmailExtractor(FieldExtractor):
    """Strategy for extracting email addresses using regular expressions."""

    def extract(self, text: str) -> str:
        """
        Scans text for the first valid email address pattern.

        Args:
            text (str): The raw resume text.

        Returns:
            str: The found email address or 'Not Found'.
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid or empty text provided for email extraction.")
            return "Not Found"

        # Improved regex for standard email formats
        pattern: str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

        try:
            match = re.search(pattern, text)
            if match:
                email: str = match.group(0)
                # Basic validation: check length and common invalid patterns
                if len(email) > 254:  # RFC 5321 limit
                    logger.warning(f"Extracted email too long: {email}")
                    return "Not Found"
                if email.count("@") != 1:
                    logger.warning(f"Invalid email format: {email}")
                    return "Not Found"
                logger.info(f"Email successfully extracted: {email}")
                return email

            logger.warning("No email address pattern found in the provided text.")
            return "Not Found"
        except Exception as e:
            logger.error(f"Unexpected error during email extraction: {e}")
            return "Error during extraction"
