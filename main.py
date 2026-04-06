import os
import json
import logging
from dotenv import load_dotenv
from core.coordinator import ResumeParserFramework, ResumeExtractor
from parsers.pdf_parser import PDFParser
from parsers.docx_parser import WordParser
from extractors.text_strategies import EmailExtractor
from extractors.llm_strategies import NameExtractor, SkillsExtractor

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    logging.warning(f"Failed to load .env file: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("resume_parser.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Runs the resume parsing pipeline for documents found in the input directory."""
    try:
        # Strategy configuration (Dependency Injection)
        parsers = {".pdf": PDFParser(), ".docx": WordParser()}

        # Name and Skills use LLM; Email uses Regex
        extractors = {
            "name": NameExtractor(),
            "email": EmailExtractor(),
            "skills": SkillsExtractor(),
        }

        coordinator = ResumeExtractor(extractors)
        framework = ResumeParserFramework(parsers, coordinator)

        INPUT_DIR = "data/resumes"
        OUTPUT_DIR = "data/output"

        # Ensure input directory exists
        if not os.path.exists(INPUT_DIR):
            logger.error(f"Input directory '{INPUT_DIR}' does not exist.")
            return

        # Create output directory with error handling
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory '{OUTPUT_DIR}': {e}")
            return

        logger.info(f"Starting resume processing in {INPUT_DIR}...")

        # List files with error handling
        try:
            files = [
                f
                for f in os.listdir(INPUT_DIR)
                if f.lower().endswith((".pdf", ".docx"))
            ]
        except OSError as e:
            logger.error(f"Failed to list files in '{INPUT_DIR}': {e}")
            return

        if not files:
            logger.warning("No resumes found to process.")
            return

        processed_count = 0
        failed_count = 0

        for filename in files:
            input_path = os.path.join(INPUT_DIR, filename)
            try:
                logger.info(f"Processing: {filename}")
                data = framework.parse_resume(input_path)

                output_path = os.path.join(
                    OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json"
                )
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(data.to_dict(), f, indent=4, ensure_ascii=False)
                    logger.info(f"Successfully exported results for {filename}")
                    processed_count += 1
                except (OSError, TypeError) as e:
                    logger.error(f"Failed to write output for {filename}: {e}")
                    failed_count += 1

            except Exception as e:
                logger.error(
                    f"Critical error processing {filename}: {e}", exc_info=True
                )
                failed_count += 1

        logger.info(
            f"Processing complete. Successfully processed: {processed_count}, "
            f"Failed: {failed_count}"
        )

    except Exception as e:
        logger.critical(f"Unexpected error in main function: {e}", exc_info=True)


if __name__ == "__main__":
    main()
