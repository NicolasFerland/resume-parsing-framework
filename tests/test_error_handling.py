"""
Error handling and edge case tests for the resume parsing framework.

This module tests how the system handles various error conditions and edge cases,
ensuring robust operation under adverse conditions. Tests cover file system errors,
parsing failures, API issues, and malformed inputs.
"""

import pytest
from core.coordinator import ResumeParserFramework
import logging
from extractors.text_strategies import EmailExtractor
from parsers.pdf_parser import PDFParser

def test_email_extractor_exception(mocker):
    """Test EmailExtractor regex failure handling."""
    extractor = EmailExtractor()
    # Force an error during re.search
    mocker.patch('extractors.text_strategies.re.search', side_effect=Exception("Regex Error"))
    result = extractor.extract("test@test.com")
    assert result == "Error during extraction"

def test_pdf_parser_file_not_found():
    """Test PDFParser handling of non-existent files."""
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.extract_text("non_existent_file.pdf")

def test_unsupported_extension(mocker):
    """Test framework rejection of unsupported file extensions."""
    """Test framework rejection of unsupported file extensions."""
    # Mock a parser so the constructor doesn't fail
    mock_parser = mocker.MagicMock()
    mock_coord = mocker.MagicMock()
    
    framework = ResumeParserFramework(
        parsers={'.pdf': mock_parser}, 
        extractor_coordinator=mock_coord
    )
    
    # Mock os.path.exists so it doesn't fail there first
    mocker.patch('core.coordinator.os.path.exists', return_value=True)
    
    with pytest.raises(ValueError, match="No parser registered"):
        framework.parse_resume("resume.docx")

def test_llm_initialization_failure(mocker):
    """Test LLM extractor graceful degradation when API key is invalid."""
    from extractors.llm_strategies import NameExtractor
    from google.auth.exceptions import DefaultCredentialsError
    
    mocker.patch('os.getenv', return_value="invalid_key")
    mocker.patch('extractors.llm_strategies.genai.configure', side_effect=DefaultCredentialsError("Invalid key"))
    
    extractor = NameExtractor()
    assert extractor.model is None
    result = extractor.extract("Some text")
    assert result == "Unknown (LLM not initialized)"

def test_parser_corruption_error(mocker):
    """Test handling of corrupted document files."""
    from parsers.docx_parser import WordParser
    
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.access', return_value=True)
    mocker.patch('docx2txt.process', side_effect=Exception("Corrupted document structure"))
    
    parser = WordParser()
    with pytest.raises(RuntimeError, match="Could not read DOCX"):
        parser.extract_text("corrupted.docx")

def test_empty_extractors_configuration():
    """Test framework validation of empty extractors configuration."""
    from core.coordinator import ResumeExtractor
    
    with pytest.raises(ValueError, match="At least one extractor must be provided"):
        ResumeExtractor({})