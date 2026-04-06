# Pluggable Resume Parsing Framework

A highly extensible Python framework designed to extract structured data (Name, Email, Skills) from various resume formats using Object-Oriented Design principles.

## Architecture & Design Patterns
- **Strategy Pattern**: Field extraction logic is encapsulated into strategy classes, allowing the system to switch between Regex, Rule-based, and LLM-based extraction at runtime.
- **Factory/Dependency Injection**: The framework is composed of injected parsers and extractors, making it easy to swap implementations (e.g., swapping `pypdf` for `pdfminer`).
- **Interface Segregation**: Strict adherence to Abstract Base Classes (ABCs) ensures that new format parsers or field extractors are consistent.

## Project Structure
- `core/`: High-level abstractions and the orchestration engine.
- `parsers/`: Logic for handling file formats (.pdf, .docx).
- `extractors/`: Specific strategies for data extraction.
- `tests/`: Unit and Integration testing suite.

## Setup
1. Create the environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   pip install -r requirements.txt
   ```

2. Run the example:
   ```bash
    python main.py
    ```

3. Run tests:

   ```bash
    python -m pytest --cov=core --cov=extractors --cov=parsers tests/
    ```

4. Generate an HTML coverage report:

   ```bash
    python -m pytest --cov=. --cov-report=html
    ```

## Testing Suite (`tests/`)
We split tests into **Unit Tests** (testing components in isolation with mocks) and **Integration Tests** (testing the full flow).

## How to Extend the Framework

This framework is built on the **Open/Closed Principle**. You can add new capabilities without modifying existing core logic.

### 1. Adding a New File Format (e.g., .txt)
1. Create a new class in `parsers/txt_parser.py` that inherits from `FileParser`.
2. Implement `extract_text(self, file_path: str) -> str`.
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