"""
Unit tests for the resume parsing framework components.

This module contains comprehensive unit tests covering all major components:
- PDF and DOCX parsers
- Email, name, and skills extractors
- Core coordinator classes
- Data models and validation

Tests use mocking to isolate components and ensure reliable CI/CD execution.
"""

import pytest
from core.models import ResumeData
from core.coordinator import ResumeExtractor
from extractors.text_strategies import EmailExtractor
from extractors.llm_strategies import NameExtractor, SkillsExtractor
import os
from io import BytesIO
from parsers.pdf_parser import PDFParser
from parsers.docx_parser import WordParser
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

@pytest.fixture(autouse=True)
def mock_genai_exceptions(mocker):
    mocker.patch('google.generativeai.exceptions', create=True)

def test_pdf_parser_real_extraction(mocker):
    """Test successful PDF text extraction with mocked pypdf library."""
    # Mocking os.path.exists is now required because of your new parser logic
    mock_reader = mocker.patch('parsers.pdf_parser.pypdf.PdfReader')
    mock_reader.return_value.is_encrypted = False
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = "PDF Content"
    mock_reader.return_value.pages = [mock_page]
    
    parser = PDFParser()
    mocker.patch('builtins.open', mocker.mock_open())
    result = parser.extract_text("fake.pdf")
    assert result == "PDF Content"

def test_docx_parser_real_extraction(mocker):
    """Test successful DOCX text extraction with mocked docx2txt library."""
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.access', return_value=True)
    mock_process = mocker.patch('docx2txt.process')
    mock_process.return_value = "Word Content"
    
    parser = WordParser()
    result = parser.extract_text("fake.docx")
    assert result == "Word Content"

def test_pdf_parser_none_page_content(mocker):
    """Test PDF parser handling when a page returns None for extract_text."""
    mock_reader = mocker.patch('parsers.pdf_parser.pypdf.PdfReader')
    mock_reader.return_value.is_encrypted = False
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = None 
    mock_reader.return_value.pages = [mock_page]
    
    parser = PDFParser()
    mocker.patch('builtins.open', mocker.mock_open())
    result = parser.extract_text("fake.pdf")
    assert result == ""

def test_pdf_parser_runtime_error(mocker):
    """Test PDF parser exception handling for corrupted files."""
    mocker.patch('parsers.pdf_parser.os.path.exists', return_value=True)
    mocker.patch('parsers.pdf_parser.os.access', return_value=True)
    # Trigger the pypdf error branch for coverage
    mocker.patch('builtins.open', side_effect=Exception("Corrupted File"))
    parser = PDFParser()
    with pytest.raises(RuntimeError, match="Could not read PDF"):
        parser.extract_text("corrupted.pdf")

def test_llm_initialization_no_key(mocker):
    """Test LLM extractor initialization when API key is not set."""
    mocker.patch('extractors.llm_strategies.os.getenv', return_value=None)
    extractor = NameExtractor()
    assert extractor.model is None

def test_llm_list_models_failure(mocker):
    """Test LLM initialization when model listing fails."""
    mocker.patch('extractors.llm_strategies.os.getenv', return_value="fake_key")
    mocker.patch('extractors.llm_strategies.genai.list_models', side_effect=Exception("API Error"))
    
    # This triggers the logger.warning or print initialization warning
    extractor = NameExtractor()
    assert extractor.model is None

def test_resume_data_model():
    """Test ResumeData dataclass creation and serialization."""
    data = ResumeData(name="John", email="test@test.com", skills=["Python"])
    assert data.name == "John"
    assert data.to_dict()["email"] == "test@test.com"

def test_email_extractor_logic():
    """Test basic email extraction from text using regex."""
    extractor = EmailExtractor()
    text = "Contact me at hello@world.com for more info"
    assert extractor.extract(text) == "hello@world.com"

def test_name_extractor_logic(mocker):
    """Test name extraction using mocked LLM response."""
    # Patch the model inside the extractor
    mock_model = mocker.patch('extractors.llm_strategies.genai.GenerativeModel')
    mock_response = mocker.MagicMock()
    mock_response.text = "Jane Doe"
    mock_model.return_value.generate_content.return_value = mock_response

    extractor = NameExtractor()
    extractor.model = mock_model.return_value  # Force the mock model
    
    text = "Jane Doe\nSoftware Engineer"
    assert extractor.extract(text) == "Jane Doe"

def test_llm_extractor_mocked(mocker):
    """Test skills extraction with mocked LLM returning comma-separated skills."""
    # This "patches" the generative model so it doesn't call the internet
    mock_model = mocker.patch('extractors.llm_strategies.genai.GenerativeModel')
    mock_response = mocker.MagicMock()
    mock_response.text = "Python, Docker, AWS"
    mock_model.return_value.generate_content.return_value = mock_response

    extractor = SkillsExtractor()
    # We bypass the 'if not self.model' check by manually assigning the mock
    extractor.model = mock_model.return_value
    
    result = extractor.extract("This is a long resume text...")
    
    assert result == ["Python", "Docker", "AWS"]
    # Verify the API was "called" exactly once (locally)
    mock_model.return_value.generate_content.assert_called_once()

def test_email_extractor_empty_text():
    """Edge Case: Ensure it doesn't crash on empty input."""
    extractor = EmailExtractor()
    assert extractor.extract("") == "Not Found"

def test_email_extractor_multiple_emails():
    """Edge Case: Ensure it picks the first email if multiple exist."""
    extractor = EmailExtractor()
    text = "Contact info@work.com or personal@home.com"
    assert extractor.extract(text) == "info@work.com"

def test_name_extractor_llm_failure(mocker):
    """Test name extractor handling of LLM API failures."""
    mock_model = mocker.patch('extractors.llm_strategies.genai.GenerativeModel')
    mock_model.return_value.generate_content.side_effect = Exception("API Timeout")
    
    extractor = NameExtractor()
    extractor.model = mock_model.return_value
    
    # Should return fallback 'Unknown' instead of crashing
    assert extractor.extract("Jane Doe") == "Unknown"

def test_resume_data_immutability():
    """Test that ResumeData instances cannot be modified after creation."""
    data = ResumeData(name="Test", email="test@test.com", skills=[])
    with pytest.raises(AttributeError):
        data.name = "New Name"

def test_pdf_parser_none_page_content(mocker):
    """Cover the branch where a PDF page returns None instead of text."""
    mock_reader = mocker.patch('pypdf.PdfReader')
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = None  # Simulate unreadable page
    mock_reader.return_value.pages = [mock_page]
    
    parser = PDFParser()
    mocker.patch('builtins.open', mocker.mock_open())
    result = parser.extract_text("fake.pdf")
    assert result == ""

def test_pdf_parser_runtime_error(mocker):
    """Cover the generic Exception/RuntimeError catch-all."""
    # Force a generic exception during file opening
    mocker.patch('builtins.open', side_effect=Exception("Corrupted File"))
    parser = PDFParser()
    with pytest.raises(RuntimeError, match="Could not read PDF"):
        parser.extract_text("corrupted.pdf")

def test_llm_initialization_fallback(mocker):
    """Cover the fallback logic when gemini-1.5-flash is missing."""
    mocker.patch('os.getenv', return_value="fake_key")
    # Mock list_models to NOT include flash, but include pro
    mock_model_pro = mocker.MagicMock()
    mock_model_pro.name = 'models/gemini-pro'
    mock_model_pro.supported_generation_methods = ['generateContent']
    
    mocker.patch('google.generativeai.list_models', return_value=[mock_model_pro])
    
    extractor = NameExtractor()
    assert extractor.model is not None
    # Verify it picked the fallback
    mock_model_name = extractor.model.model_name
    assert 'gemini-pro' in mock_model_name

def test_llm_extraction_exception(mocker):
    """Cover the try-except block in the extract method."""
    mock_gen = mocker.patch('extractors.llm_strategies.genai.GenerativeModel')
    # Force the generate_content call to crash
    mock_gen.return_value.generate_content.side_effect = Exception("Quota Exceeded")
    
    extractor = SkillsExtractor()
    extractor.model = mock_gen.return_value
    result = extractor.extract("Some resume text")
    assert any("Extraction failed" in s or "Extraction Error" in s for s in result)

def test_name_extractor_api_crash(mocker):
    """Test name extractor handling of API server crashes."""
    # 1. Initialize extractor
    extractor = NameExtractor()
    
    # 2. Mock the model and force a crash on generate_content
    # We patch the class to ensure any new instance uses this mock
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = Exception("API Server Down")
    
    # MANUALLY assign the mock to this specific instance
    extractor.model = mock_model
    
    # 3. Execute and verify the fallback return
    result = extractor.extract("John Doe")
    assert result == "Unknown"

def test_skills_extractor_empty_response_text(mocker):
    """Test skills extractor handling of empty LLM response text."""
    extractor = SkillsExtractor()
    mock_model = mocker.MagicMock()
    mock_res = mocker.MagicMock()
    
    # Set text to an empty string to trigger line 65
    mock_res.text = ""  
    mock_model.generate_content.return_value = mock_res
    
    # MANUALLY assign the mock to this specific instance
    extractor.model = mock_model
    
    result = extractor.extract("Some text")
    assert result == []

def test_skills_extractor_api_crash(mocker):
    """Test skills extractor handling of API connection timeouts."""
    extractor = SkillsExtractor()
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = Exception("Connection Timeout")
    extractor.model = mock_model
    
    result = extractor.extract("Python, Java")
    # Verify the error message is caught and returned as a list item
    assert any("Extraction Error" in s for s in result)

def test_name_extractor_not_initialized(mocker):
    """Test name extractor when LLM is not initialized."""
    mocker.patch('os.getenv', return_value=None)
    extractor = NameExtractor()
    # Ensure model is None to trigger the guard clause
    extractor.model = None 
    
    result = extractor.extract("Some text")
    assert result == "Unknown (LLM not initialized)"

def test_skills_extractor_not_initialized(mocker):
    """Test skills extractor when LLM is not initialized."""
    mocker.patch('os.getenv', return_value=None)
    extractor = SkillsExtractor()
    # Ensure model is None to trigger the guard clause
    extractor.model = None
    
    result = extractor.extract("Some text")
    assert result == ["Error: LLM not initialized"]

def test_coordinator_missing_extractors(mocker):
    """Test coordinator behavior when some extractors are missing from configuration."""
    # Provide only 'name', leaving 'email' and 'skills' missing to trigger KeyErrors
    extractors = {'name': mocker.MagicMock()}
    extractors['name'].extract.return_value = "Test Name"
    
    coordinator = ResumeExtractor(extractors)
    # Trigger orchestration with missing keys
    result = coordinator.orchestrate("Valid text")
    
    assert result.name == "Test Name"
    assert result.email == "Not Found" # Triggered by line 48-53
    assert result.skills == []         # Triggered by line 57-62

# Test for encrypted PDFs removed as encrypted check was removed from parser

def test_skills_extractor_quota_error(mocker):
    """Test skills extractor handling of API quota exhaustion."""
    mock_gen = mocker.patch('extractors.llm_strategies.genai.GenerativeModel')
    # Use the mock exception we created in Step 1
    mock_gen.return_value.generate_content.side_effect = Exception("API quota exceeded")
    
    extractor = SkillsExtractor()
    extractor.model = mock_gen.return_value
    result = extractor.extract("text")
    assert "API quota exceeded" in result[0] # Covers line 138-140

# ============== COMPREHENSIVE COVERAGE TESTS ==============

# core/coordinator.py - Line 20: Empty extractors validation
def test_coordinator_empty_extractors():
    """Test that ResumeExtractor raises ValueError with empty extractors dict."""
    with pytest.raises(ValueError, match="At least one extractor must be provided"):
        ResumeExtractor({})

# core/coordinator.py - Name extraction KeyError
def test_coordinator_name_key_error(mocker):
    """Test KeyError when name extractor is missing from dict."""
    # Provide email and skills but NOT name
    mock_email = mocker.MagicMock()
    mock_email.extract.return_value = "test@test.com"
    mock_skills = mocker.MagicMock()
    mock_skills.extract.return_value = ["Python"]
    
    coordinator = ResumeExtractor({'email': mock_email, 'skills': mock_skills})
    result = coordinator.orchestrate("Valid text")
    
    assert result.name == "Unknown"
    assert result.email == "test@test.com"
    assert result.skills == ["Python"]

# core/coordinator.py - Email extraction KeyError
def test_coordinator_email_key_error(mocker):
    """Test KeyError when email extractor is missing from dict."""
    mock_name = mocker.MagicMock()
    mock_name.extract.return_value = "John Doe"
    mock_skills = mocker.MagicMock()
    mock_skills.extract.return_value = ["Python"]
    
    coordinator = ResumeExtractor({'name': mock_name, 'skills': mock_skills})
    result = coordinator.orchestrate("Valid text")
    
    assert result.name == "John Doe"
    assert result.email == "Not Found"
    assert result.skills == ["Python"]

# core/coordinator.py - Skills extraction KeyError
def test_coordinator_skills_key_error(mocker):
    """Test KeyError when skills extractor is missing from dict."""
    mock_name = mocker.MagicMock()
    mock_name.extract.return_value = "Jane Doe"
    mock_email = mocker.MagicMock()
    mock_email.extract.return_value = "jane@test.com"
    
    coordinator = ResumeExtractor({'name': mock_name, 'email': mock_email})
    result = coordinator.orchestrate("Valid text")
    
    assert result.name == "Jane Doe"
    assert result.email == "jane@test.com"
    assert result.skills == []

# core/coordinator.py - Empty text returned from parser
def test_resume_parser_framework_empty_text(mocker):
    """Test parse_resume when parser returns empty text."""
    from core.coordinator import ResumeParserFramework
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp_path = tmp.name
    
    try:
        mock_parser = mocker.MagicMock()
        mock_parser.extract_text.return_value = ""  # Empty text
        
        mock_coordinator = mocker.MagicMock()
        mock_coordinator.orchestrate.return_value = ResumeData(name="Unknown", email="Not Found", skills=[])
        
        framework = ResumeParserFramework({'.pdf': mock_parser}, mock_coordinator)
        result = framework.parse_resume(tmp_path)
        
        assert result.name == "Unknown"
    finally:
        os.unlink(tmp_path)

# core/coordinator.py - Line 34-35: Invalid text in orchestrate
def test_coordinator_none_text(mocker):
    """Test orchestrate with None text input."""
    extractor = ResumeExtractor({'name': mocker.MagicMock()})
    result = extractor.orchestrate(None)
    assert result.name == "Unknown"
    assert result.email == "Not Found"
    assert result.skills == []

def test_coordinator_non_string_text(mocker):
    """Test orchestrate with non-string text input."""
    extractor = ResumeExtractor({'name': mocker.MagicMock()})
    result = extractor.orchestrate(123)
    assert result.name == "Unknown"
    assert result.email == "Not Found"
    assert result.skills == []

def test_coordinator_empty_string_text(mocker):
    """Test orchestrate with empty string text input."""
    extractor = ResumeExtractor({'name': mocker.MagicMock()})
    result = extractor.orchestrate("")
    assert result.name == "Unknown"
    assert result.email == "Not Found"
    assert result.skills == []

# core/coordinator.py - Name extraction exception handling
def test_coordinator_name_exception(mocker):
    """Test exception during name extraction."""
    mock_name = mocker.MagicMock()
    mock_name.extract.side_effect = Exception("Name extraction error")
    mock_email = mocker.MagicMock()
    mock_email.extract.return_value = "test@test.com"
    mock_skills = mocker.MagicMock()
    mock_skills.extract.return_value = []
    
    extractor = ResumeExtractor({'name': mock_name, 'email': mock_email, 'skills': mock_skills})
    result = extractor.orchestrate("Some text")
    assert result.name == "Unknown"
    assert result.email == "test@test.com"

# core/coordinator.py - Email extraction exception handling
def test_coordinator_email_exception(mocker):
    """Test exception during email extraction."""
    mock_name = mocker.MagicMock()
    mock_name.extract.return_value = "John Doe"
    mock_email = mocker.MagicMock()
    mock_email.extract.side_effect = Exception("Email extraction error")
    mock_skills = mocker.MagicMock()
    mock_skills.extract.return_value = []
    
    extractor = ResumeExtractor({'name': mock_name, 'email': mock_email, 'skills': mock_skills})
    result = extractor.orchestrate("Some text")
    assert result.name == "John Doe"
    assert result.email == "Not Found"

# core/coordinator.py - Skills extraction exception handling
def test_coordinator_skills_exception(mocker):
    """Test exception during skills extraction."""
    mock_name = mocker.MagicMock()
    mock_name.extract.return_value = "Jane Doe"
    mock_email = mocker.MagicMock()
    mock_email.extract.return_value = "jane@test.com"
    mock_skills = mocker.MagicMock()
    mock_skills.extract.side_effect = Exception("Skills extraction error")
    
    extractor = ResumeExtractor({'name': mock_name, 'email': mock_email, 'skills': mock_skills})
    result = extractor.orchestrate("Some text")
    assert result.email == "jane@test.com"
    assert result.skills == []

# core/coordinator.py - ResumeParserFramework tests
def test_resume_parser_framework_invalid_path(mocker):
    """Test parse_resume with invalid file path."""
    from core.coordinator import ResumeParserFramework
    framework = ResumeParserFramework({}, mocker.MagicMock())
    with pytest.raises(ValueError, match="Invalid file path"):
        framework.parse_resume(None)

def test_resume_parser_framework_file_not_found(mocker):
    """Test parse_resume with non-existent file."""
    from core.coordinator import ResumeParserFramework
    framework = ResumeParserFramework({}, mocker.MagicMock())
    with pytest.raises(FileNotFoundError):
        framework.parse_resume("/nonexistent/file.pdf")

def test_resume_parser_framework_no_extension(mocker):
    """Test parse_resume with file path without extension."""
    from core.coordinator import ResumeParserFramework
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='') as tmp:
        tmp_path = tmp.name
    
    try:
        framework = ResumeParserFramework({}, mocker.MagicMock())
        with pytest.raises(ValueError, match="No file extension found"):
            framework.parse_resume(tmp_path)
    finally:
        os.unlink(tmp_path)

def test_resume_parser_framework_unsupported_extension(mocker):
    """Test parse_resume with unsupported file extension."""
    from core.coordinator import ResumeParserFramework
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
        tmp_path = tmp.name
    
    try:
        framework = ResumeParserFramework({}, mocker.MagicMock())
        with pytest.raises(ValueError, match="No parser registered"):
            framework.parse_resume(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_resume_parser_framework_parse_error(mocker):
    """Test parse_resume with parser that raises exception."""
    from core.coordinator import ResumeParserFramework
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp_path = tmp.name
    
    try:
        mock_parser = mocker.MagicMock()
        mock_parser.extract_text.side_effect = Exception("Parser error")
        
        mock_coordinator = mocker.MagicMock()
        framework = ResumeParserFramework({'.pdf': mock_parser}, mock_coordinator)
        
        with pytest.raises(Exception, match="Parser error"):
            framework.parse_resume(tmp_path)
    finally:
        os.unlink(tmp_path)

# extractors/llm_strategies.py - No available models
def test_llm_no_available_models(mocker):
    """Test LLM initialization when no models support generateContent."""
    mocker.patch('os.getenv', return_value="fake_key")
    mock_model = mocker.MagicMock()
    mock_model.supported_generation_methods = ['embedContent']  # No generateContent
    mocker.patch('extractors.llm_strategies.genai.list_models', return_value=[mock_model])
    
    extractor = NameExtractor()
    assert extractor.model is None

def test_llm_no_suitable_gemini_model(mocker):
    """Test LLM fallback when no gemini model found."""
    mocker.patch('os.getenv', return_value="fake_key")
    mock_model = mocker.MagicMock()
    mock_model.name = 'models/other-model'
    mock_model.supported_generation_methods = ['generateContent']
    mocker.patch('extractors.llm_strategies.genai.list_models', return_value=[mock_model])
    
    extractor = NameExtractor()
    # Should still be None as next() returns None when no match
    assert extractor.model is None

def test_llm_default_credentials_error(mocker):
    """Test LLM handling of DefaultCredentialsError."""
    from google.auth.exceptions import DefaultCredentialsError
    mocker.patch('os.getenv', return_value="invalid_key")
    mocker.patch('extractors.llm_strategies.genai.configure', side_effect=DefaultCredentialsError("Auth error"))
    
    extractor = NameExtractor()
    assert extractor.model is None

# extractors/llm_strategies.py - NameExtractor empty text
def test_name_extractor_empty_text(mocker):
    """Test NameExtractor with empty text."""
    mock_model = mocker.MagicMock()
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("")
    assert result == "Unknown"

def test_name_extractor_whitespace_only(mocker):
    """Test NameExtractor with whitespace-only text."""
    mock_model = mocker.MagicMock()
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("   \n  \t  ")
    assert result == "Unknown"

# extractors/llm_strategies.py - NameExtractor empty response
def test_name_extractor_empty_response(mocker):
    """Test NameExtractor with empty response from LLM."""
    mock_model = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.text = ""
    mock_model.generate_content.return_value = mock_response
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == "Unknown"

def test_name_extractor_none_response(mocker):
    """Test NameExtractor with None response from LLM."""
    mock_model = mocker.MagicMock()
    mock_model.generate_content.return_value = None
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == "Unknown"

# extractors/llm_strategies.py - NameExtractor ResourceExhausted
def test_name_extractor_resource_exhausted(mocker):
    """Test NameExtractor with ResourceExhausted exception."""
    from google.api_core.exceptions import ResourceExhausted
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = ResourceExhausted("API quota exceeded")
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert "API quota exceeded" in result

# extractors/llm_strategies.py - NameExtractor ServiceUnavailable
def test_name_extractor_service_unavailable(mocker):
    """Test NameExtractor with ServiceUnavailable exception."""
    from google.api_core.exceptions import ServiceUnavailable
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = ServiceUnavailable("Service down")
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert "Service unavailable" in result

# extractors/llm_strategies.py - NameExtractor returns "Unknown" from LLM
def test_name_extractor_unknown_response(mocker):
    """Test NameExtractor when LLM response is 'Unknown'."""
    mock_model = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.text = "Unknown"  # LLM returned Unknown
    mock_model.generate_content.return_value = mock_response
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == "Unknown"

# extractors/llm_strategies.py - NameExtractor empty text
def test_name_extractor_empty_text(mocker):
    """Test NameExtractor with empty text."""
    mock_model = mocker.MagicMock()
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("")
    assert result == "Unknown"

def test_name_extractor_whitespace_only(mocker):
    """Test NameExtractor with whitespace-only text."""
    mock_model = mocker.MagicMock()
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("   \n  \t  ")
    assert result == "Unknown"

# extractors/llm_strategies.py - NameExtractor empty response
def test_name_extractor_empty_response(mocker):
    """Test NameExtractor with empty response from LLM."""
    mock_model = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.text = ""
    mock_model.generate_content.return_value = mock_response
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == "Unknown"

def test_name_extractor_none_response(mocker):
    """Test NameExtractor with None response from LLM."""
    mock_model = mocker.MagicMock()
    mock_model.generate_content.return_value = None
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == "Unknown"

# extractors/llm_strategies.py - NameExtractor ResourceExhausted
def test_name_extractor_resource_exhausted(mocker):
    """Test NameExtractor with ResourceExhausted exception."""
    from google.api_core.exceptions import ResourceExhausted
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = ResourceExhausted("API quota exceeded")
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert "API quota exceeded" in result

# extractors/llm_strategies.py - NameExtractor ServiceUnavailable
def test_name_extractor_service_unavailable(mocker):
    """Test NameExtractor with ServiceUnavailable exception."""
    from google.api_core.exceptions import ServiceUnavailable
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = ServiceUnavailable("Service down")
    
    extractor = NameExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert "Service unavailable" in result

# extractors/llm_strategies.py - SkillsExtractor returns empty from response
def test_skills_extractor_empty_response_from_llm(mocker):
    """Test SkillsExtractor when LLM returns empty string with commas."""
    mock_model = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.text = ", , , "  # Just commas, no actual skills
    mock_model.generate_content.return_value = mock_response
    
    extractor = SkillsExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == []

# extractors/llm_strategies.py - SkillsExtractor empty text
def test_skills_extractor_empty_text(mocker):
    """Test SkillsExtractor with empty text."""
    mock_model = mocker.MagicMock()
    extractor = SkillsExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("")
    assert result == []

def test_skills_extractor_whitespace_only(mocker):
    """Test SkillsExtractor with whitespace-only text."""
    mock_model = mocker.MagicMock()
    extractor = SkillsExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("   \n  \t  ")
    assert result == []

# extractors/llm_strategies.py - SkillsExtractor empty response
def test_skills_extractor_empty_response(mocker):
    """Test SkillsExtractor with empty response from LLM."""
    mock_model = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.text = ""
    mock_model.generate_content.return_value = mock_response
    
    extractor = SkillsExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == []

def test_skills_extractor_none_response(mocker):
    """Test SkillsExtractor with None response from LLM."""
    mock_model = mocker.MagicMock()
    mock_model.generate_content.return_value = None
    
    extractor = SkillsExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert result == []

# extractors/llm_strategies.py - SkillsExtractor ResourceExhausted
def test_skills_extractor_resource_exhausted(mocker):
    """Test SkillsExtractor with ResourceExhausted exception."""
    from google.api_core.exceptions import ResourceExhausted
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = ResourceExhausted("API quota exceeded")
    
    extractor = SkillsExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert "API quota exceeded" in result[0]

# extractors/llm_strategies.py - SkillsExtractor ServiceUnavailable
def test_skills_extractor_service_unavailable(mocker):
    """Test SkillsExtractor with ServiceUnavailable exception."""
    from google.api_core.exceptions import ServiceUnavailable
    mock_model = mocker.MagicMock()
    mock_model.generate_content.side_effect = ServiceUnavailable("Service down")
    
    extractor = SkillsExtractor()
    extractor.model = mock_model
    
    result = extractor.extract("Some resume text")
    assert "Service unavailable" in result[0]

# extractors/text_strategies.py - Email too long
def test_email_extractor_too_long(mocker):
    """Test EmailExtractor with email exceeding RFC 5321 limit."""
    extractor = EmailExtractor()
    long_email = "a" * 250 + "@test.com"  # > 254 chars
    text = f"Contact: {long_email}"
    
    result = extractor.extract(text)
    assert result == "Not Found"

# extractors/text_strategies.py - Email with exactly 1 at but not passing validation
def test_email_extractor_at_count_edge_case(mocker):
    """Test EmailExtractor validation of @ count."""
    extractor = EmailExtractor()
    # Create a scenario where email.count('@') returns something other than 1
    # This requires mocking the email after extraction
    text = "test@example.com"
    
    # Mock the part after extraction to simulate multiple @ being found
    original_search = mocker.patch('extractors.text_strategies.re.search')
    mock_match = mocker.MagicMock()
    mock_match.group.return_value = "test@example@com"  # Manually set to have 2 @
    original_search.return_value = mock_match
    
    result = extractor.extract(text)
    assert result == "Not Found"

# extractors/text_strategies.py - Exception in email extraction
def test_email_extractor_exception(mocker):
    """Test EmailExtractor exception handling."""
    extractor = EmailExtractor()
    # Mock re.search to raise an exception
    mocker.patch('extractors.text_strategies.re.search', side_effect=Exception("Regex error"))
    
    result = extractor.extract("test@test.com")
    assert result == "Error during extraction"

# extractors/text_strategies.py - No email pattern
def test_email_extractor_no_pattern(mocker):
    """Test EmailExtractor with no email pattern in text."""
    extractor = EmailExtractor()
    text = "This resume has no email address at all"
    
    result = extractor.extract(text)
    assert result == "Not Found"

# parsers/docx_parser.py - None text from docx2txt
def test_docx_parser_none_text(mocker):
    """Test WordParser when docx2txt.process returns None."""
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.access', return_value=True)
    mocker.patch('docx2txt.process', return_value=None)
    
    parser = WordParser()
    result = parser.extract_text("test.docx")
    assert result == ""

# parsers/docx_parser.py - Exception in docx_parser
def test_docx_parser_exception(mocker):
    """Test WordParser exception handling."""
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.access', return_value=True)
    mocker.patch('docx2txt.process', side_effect=Exception("Corrupted DOCX"))
    
    parser = WordParser()
    with pytest.raises(RuntimeError, match="Could not read DOCX"):
        parser.extract_text("corrupted.docx")

# parsers/pdf_parser.py - Page with no extractable text
def test_pdf_parser_page_no_text(mocker):
    """Test PDFParser when a page returns no text."""
    mock_reader = mocker.patch('pypdf.PdfReader')
    mock_reader.return_value.is_encrypted = False
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = None
    mock_reader.return_value.pages = [mock_page]
    
    parser = PDFParser()
    mocker.patch('builtins.open', mocker.mock_open())
    result = parser.extract_text("test.pdf")
    assert result == ""

def test_pdf_parser_page_empty_string(mocker):
    """Test PDFParser when a page returns empty string."""
    mock_reader = mocker.patch('pypdf.PdfReader')
    mock_reader.return_value.is_encrypted = False
    mock_page1 = mocker.MagicMock()
    mock_page1.extract_text.return_value = ""
    mock_page2 = mocker.MagicMock()
    mock_page2.extract_text.return_value = "Some text"
    mock_reader.return_value.pages = [mock_page1, mock_page2]
    
    parser = PDFParser()
    mocker.patch('builtins.open', mocker.mock_open())
    result = parser.extract_text("test.pdf")
    assert "Some text" in result

# parsers/pdf_parser.py - Page extraction exception
def test_pdf_parser_page_extraction_error(mocker):
    """Test PDFParser when page extraction raises exception."""
    mock_reader = mocker.patch('pypdf.PdfReader')
    mock_reader.return_value.is_encrypted = False
    mock_page = mocker.MagicMock()
    mock_page.extract_text.side_effect = Exception("Page read error")
    mock_reader.return_value.pages = [mock_page]
    
    parser = PDFParser()
    mocker.patch('builtins.open', mocker.mock_open())
    result = parser.extract_text("test.pdf")
    assert result == ""

# parsers/pdf_parser.py - PdfReadError
def test_pdf_parser_read_error(mocker):
    """Test PDFParser with PdfReadError."""
    mock_reader = mocker.patch('pypdf.PdfReader')
    mock_reader.side_effect = mocker.MagicMock(
        side_effect=Exception("PdfReadError")
    )
    # Mock the pypdf.errors module
    mocker.patch('pypdf.errors.PdfReadError', Exception)
    
    parser = PDFParser()
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    # Create a proper exception instance
    class MockPdfReadError(Exception):
        pass
    
    mock_reader.side_effect = MockPdfReadError("Corrupted")
    
    with pytest.raises(RuntimeError, match="Corrupted PDF"):
        parser.extract_text("corrupted.pdf")

# parsers/pdf_parser.py - FileNotFoundError
def test_pdf_parser_file_not_found(mocker):
    """Test PDFParser with FileNotFoundError."""
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.extract_text("/nonexistent/file.pdf")

# parsers/pdf_parser.py - Empty PDF
def test_pdf_parser_empty_pdf(mocker):
    """Test PDFParser with PDF that has only whitespace content."""
    mock_reader = mocker.patch('pypdf.PdfReader')
    mock_reader.return_value.is_encrypted = False
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = "   \n  \t  "
    mock_reader.return_value.pages = [mock_page]
    
    parser = PDFParser()
    mocker.patch('builtins.open', mocker.mock_open())
    result = parser.extract_text("test.pdf")
    assert result == ""