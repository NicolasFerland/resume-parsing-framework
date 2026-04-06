from core.coordinator import ResumeParserFramework, ResumeExtractor
from extractors.text_strategies import EmailExtractor
from extractors.llm_strategies import NameExtractor, SkillsExtractor


class MockPDFParser:
    """Mock parser to avoid requiring physical files during CI/CD tests."""

    def extract_text(self, file_path: str) -> str:
        # Standard clues included: Name at top, email, and skills section
        return "Alice Smith\nalice@example.com\nSkills: Python, Java"


def test_full_framework_flow(mocker):
    """
    Tests the full orchestration from parsing to multi-strategy extraction.
    Covers Happy Path: Valid text extraction and successful LLM/Regex results.
    """
    # 0. Mock file system checks
    mocker.patch("os.path.exists", return_value=True)

    # 1. Mock the GenerativeModel globally for the extractors module
    mock_gen = mocker.patch("extractors.llm_strategies.genai.GenerativeModel")

    # 2. Setup distinct responses for Name and Skills calls
    mock_name_res = mocker.MagicMock()
    mock_name_res.text = "Alice Smith"

    mock_skills_res = mocker.MagicMock()
    mock_skills_res.text = "Python, Java, SQL"

    # Configure the mock to return Name first, then Skills
    mock_gen.return_value.generate_content.side_effect = [
        mock_name_res,
        mock_skills_res,
    ]

    # 3. Setup Framework with Strategy Pattern
    parsers = {".pdf": MockPDFParser()}

    name_ext = NameExtractor()
    name_ext.model = mock_gen.return_value  # Ensure NameExtractor is initialized

    skills_ext = SkillsExtractor()
    skills_ext.model = mock_gen.return_value  # Ensure SkillsExtractor is initialized

    extractors = {"name": name_ext, "email": EmailExtractor(), "skills": skills_ext}

    coordinator = ResumeExtractor(extractors)
    framework = ResumeParserFramework(parsers, coordinator)

    # 4. Execute Extraction
    result = framework.parse_resume("fake_path.pdf")

    # 5. Assertions (Happy Path)
    # Check Name (LLM-based)
    assert result.name == "Alice Smith"
    # Check Email (Regex-based)
    assert result.email == "alice@example.com"
    # Check Skills (LLM-based)
    assert "Python" in result.skills
    assert "SQL" in result.skills

    # Verify the LLM was called exactly twice (once for name, once for skills)
    assert mock_gen.return_value.generate_content.call_count == 2
