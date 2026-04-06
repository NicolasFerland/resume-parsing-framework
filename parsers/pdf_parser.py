import pypdf
import logging
from core.interfaces import FileParser

logger = logging.getLogger(__name__)


class PDFParser(FileParser):
    """Concrete implementation for parsing PDF documents."""

    def extract_text(self, file_path: str) -> str:
        """
        Extracts all readable text from a PDF file.

        Args:
            file_path (str): Path to the .pdf file.

        Returns:
            str: Combined text from all pages.

        Raises:
            RuntimeError: If the PDF is encrypted, corrupted, or unreadable.
        """
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)

                text = ""
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                        else:
                            logger.warning(
                                f"Page {i+1} in {file_path} has no extractable text."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {i+1} in {file_path}: "
                            f"{e}"
                        )

                if not text.strip():
                    logger.warning(
                        f"PDF at {file_path} appears to be empty or " "image-based."
                    )
                    return ""

                logger.info(
                    f"Successfully extracted {len(text)} characters from {file_path}"
                )
                return text

        except pypdf.errors.PdfReadError as e:
            logger.error(f"PDF read error for {file_path}: {e}")
            raise RuntimeError(f"Corrupted PDF: {e}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing PDF {file_path}: {e}")
            raise RuntimeError(f"Could not read PDF: {e}")
