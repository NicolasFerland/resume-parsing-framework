# Pluggable Resume Parsing Framework

A highly extensible Python framework designed to extract structured data (Name, Email, Skills) from various resume formats using Object-Oriented Design principles.

## Architecture & Design Patterns
- **Strategy Pattern**: Field extraction logic is encapsulated into strategy classes, allowing the system to switch between Regex, Rule-based, and LLM-based extraction at runtime.
- **Factory/Dependency Injection**: The framework is composed of injected parsers and extractors, making it easy to swap implementations (e.g., swapping `pypdf` for `pdfminer`).
- **Interface Segregation**: Strict adherence to Abstract Base Classes (ABCs) ensures that new format parsers or field extractors are consistent.
- **Coordinator Pattern**: The `ResumeExtractor` orchestrates field extraction, while `ResumeParserFramework` coordinates file parsing and extraction.

## Project Structure
- `core/`: High-level abstractions and the orchestration engine.
- `parsers/`: Logic for handling file formats (.pdf, .docx).
- `extractors/`: Specific strategies for data extraction.
- `tests/`: Unit and Integration testing suite with 100% coverage.

## Features
- ✅ **Multiple File Formats**: Supports PDF and Word Document parsing
- ✅ **Pluggable Extractors**: Regex-based EmailExtractor, LLM-based NameExtractor and SkillsExtractor
- ✅ **Robust Error Handling**: Comprehensive exception handling with informative logging
- ✅ **Extensible Design**: Easy to add new parsers and extractors via interfaces
- ✅ **Complete Test Coverage**: 72 tests covering all happy paths and edge cases
- ✅ **Production Ready**: Type hints, docstrings, and professional code organization

## Setup & Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key (for LLM-based extraction)

### Installation
1. Clone the repository and navigate to the project directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create a .env file in the project root
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

## Usage Examples

```bash
# Run the main processing script
python main.py

# This will process all .pdf and .docx files in data/resumes/
# and output structured JSON files to data/output/
```

## Testing

### Run All Tests
```bash
python -m pytest tests/
```

### Run Tests with Coverage Report
```bash
python -m pytest --cov=core --cov=extractors --cov=parsers --cov-report=term-missing tests/
```

### Generate HTML Coverage Report
```bash
python -m pytest --cov=. --cov-report=html
# View report at htmlcov/index.html
```

**Test Coverage**: 100% across all modules (248 statements, 0 missed)

## How to Extend the Framework

This framework is built on the **Open/Closed Principle**. You can add new capabilities without modifying existing core logic.

### 1. Adding a New File Format (e.g., .txt)
1. Create a new class in `parsers/txt_parser.py` that inherits from `FileParser`.
2. Implement `extract_text(self, file_path: str) -> str`.
3. Register it in the parsers dictionary: `parsers['.txt'] = TXTParser()`

### 2. Adding a New Field Extractor (e.g., Phone Number)
1. Create a new class in `extractors/phone_strategies.py` that inherits from `FieldExtractor`.
2. Implement `extract(self, text: str) -> Any`.
3. Add it to the extractors dictionary: `extractors['phone'] = PhoneExtractor()`

### 3. Adding a New Extraction Strategy (e.g., NER-based)
1. Create a new extractor class inheriting from `FieldExtractor`.
2. Implement your NER logic in the `extract` method.
3. Use dependency injection to swap it in: `extractors['name'] = NERNameExtractor()`

## Error Handling & Logging

The framework includes comprehensive error handling:
- **File I/O Errors**: Graceful handling of corrupted or missing files
- **API Failures**: LLM quota exceeded, service unavailable, authentication errors
- **Invalid Input**: None values, empty strings, wrong file types
- **Parsing Errors**: Corrupted PDFs, DOCX files, encoding issues

All errors are logged with informative messages and the system continues processing other files.

## Dependencies

- `pypdf`: PDF text extraction
- `docx2txt`: Word document text extraction
- `google-generativeai`: LLM-based extraction (Gemini API)
- `python-dotenv`: Environment variable management
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting

## API Reference

### Core Classes

#### `ResumeData`
Data class representing extracted resume information.
- `name: str` - Candidate's full name
- `email: str` - Email address
- `skills: List[str]` - List of technical/professional skills

#### `FileParser` (Abstract Base Class)
Interface for file format parsers.
- `extract_text(file_path: str) -> str` - Extract text content from file

#### `FieldExtractor` (Abstract Base Class)
Interface for field-specific extractors.
- `extract(text: str) -> Any` - Extract specific field from text

#### `ResumeExtractor`
Orchestrates field extraction from text.
- `__init__(extractors: Dict[str, FieldExtractor])` - Initialize with extractor strategies
- `orchestrate(text: str) -> ResumeData` - Extract all fields and return structured data

#### `ResumeParserFramework`
Main framework class combining parsing and extraction.
- `__init__(parsers: Dict[str, FileParser], extractor_coordinator: ResumeExtractor)` - Initialize framework
- `parse_resume(file_path: str) -> ResumeData` - Parse file and extract structured data

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure 100% test coverage for new code

## License

This project is provided as-is for demonstration purposes of my skills.
3. Register the new parser in `main.py`:
   ```python
   parsers = {'.pdf': PDFParser(), '.docx': WordParser(), '.txt': TxtParser()}

### 2. Adding a New Field (e.g., "Experience")
1. Define the Extractor: Create ExperienceExtractor in extractors/llm_strategies.py.

2. Update the Data Model: Add experience: str to the ResumeData dataclass in models.py.

3. Update the Coordinator: Update ResumeExtractor.orchestrate to include the new field.

Proposed Extensions for Common Resume Parts
- Contact Information: A Regex-based PhoneExtractor to capture international phone formats.

- Work History: An LLM-based ExperienceExtractor that returns a list of objects containing job_title, company, and duration.

- Education: A hybrid EducationExtractor using NER (Named Entity Recognition) to identify universities and graduation years.