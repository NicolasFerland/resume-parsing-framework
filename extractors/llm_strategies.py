"""
LLM-based extraction strategies for resume parsing.

This module provides extractors that use Google's Gemini AI to identify
candidate names and skills from resume text. The extractors are designed
to handle the complexity of natural language processing that regex patterns
cannot easily capture.

Key considerations:
- API key must be set via GEMINI_API_KEY environment variable
- Context is limited to manage token costs and API quotas
- Fallback to 'Unknown' or empty results when LLM is unavailable
"""

import os
import logging
from typing import Any, List, Optional
import google.generativeai as genai
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from core.interfaces import FieldExtractor

logger = logging.getLogger(__name__)


class BaseLLMExtractor(FieldExtractor):
    """
    Base class for LLM-based extraction to manage model initialization.

    Handles the setup of Google Gemini API client and automatic model selection.
    Prioritizes gemini-1.5-flash for its balance of speed and accuracy, with
    fallback to other available Gemini models if the preferred one isn't accessible.
    Requires GEMINI_API_KEY environment variable to be set.
    """

    def __init__(self):
        """
        Initializes the LLM client and selects a supported Gemini model if available.

        Attempts to use gemini-1.5-flash as the primary model due to its good
        performance-to-cost ratio. Falls back to any available Gemini model if
        the preferred one is not accessible. If no models are available or API
        key is missing, the extractor will gracefully degrade to return default values.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        self.model: Optional[Any] = None

        if not api_key:
            logger.warning(
                "GEMINI_API_KEY environment variable not set. "
                "LLM features will be unavailable."
            )
            return

        try:
            genai.configure(api_key=api_key)
            # Get list of models that support content generation
            available_models = [
                m.name
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]

            if not available_models:
                logger.warning("No supported Gemini models available.")
                return

            # Prefer gemini-1.5-flash for its speed and cost efficiency
            selected_model = "models/gemini-1.5-flash"
            if selected_model not in available_models:
                # Fallback to any available Gemini model
                selected_model = next(
                    (m for m in available_models if "gemini" in m), None
                )

            if selected_model:
                self.model = genai.GenerativeModel(selected_model)
                logger.info(
                    f"LLM initialized successfully with model: {selected_model}"
                )
            else:
                logger.warning("No suitable Gemini model found.")

        except DefaultCredentialsError:
            logger.error("Invalid or missing API key for Gemini.")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")


class NameExtractor(BaseLLMExtractor):
    """
    LLM-based Name Extraction (Robust over simple rule-based).

    Uses AI to identify candidate names from resume text. More reliable than
    regex patterns for handling various name formats and contexts where names
    appear in headers, signatures, or other non-standard positions.
    """

    def extract(self, text: str) -> str:
        """
        Uses LLM to identify the candidate's name from resume context.

        Args:
            text (str): Raw resume text.

        Returns:
            str: The identified name or 'Unknown'.
        """
        if not self.model:
            logger.warning("LLM not initialized, cannot extract name.")
            return "Unknown (LLM not initialized)"

        if not text or not text.strip():
            logger.warning("Empty or invalid text provided for name extraction.")
            return "Unknown"

        # Limit to first 1000 characters as names typically appear early in resumes
        # This reduces token costs while maintaining high accuracy
        context = text[:1000]
        prompt = (
            "Identify the full name of the candidate from this resume text. "
            "The name is usually at the top, but look for standard clues like "
            "larger fonts or primary headers. Return ONLY the name. If not found, "
            "return 'Unknown'. "
            f"Text: {context}"
        )

        try:
            response = self.model.generate_content(prompt)
            if response and response.text:
                result = response.text.strip()
                if result and result != "Unknown":
                    logger.info(f"Name extracted: {result}")
                    return result
                else:
                    logger.info("No name found in response.")
                    return "Unknown"
            else:
                logger.warning("Empty response from LLM for name extraction.")
                return "Unknown"

        except ResourceExhausted:
            logger.error("API quota exceeded for name extraction.")
            return "Unknown (API quota exceeded)"
        except ServiceUnavailable:
            logger.error("Gemini service unavailable for name extraction.")
            return "Unknown (Service unavailable)"
        except Exception as e:
            logger.error(f"Name extraction failed: {e}")
            return "Unknown"


class SkillsExtractor(BaseLLMExtractor):
    """
    LLM-based Skills Extraction.

    Identifies both technical and soft skills from resume content using AI.
    More comprehensive than keyword matching as it can understand context
    and identify skills mentioned in various formats.
    """

    def extract(self, text: str) -> List[str]:
        """Extracts skill items from resume text using the configured LLM.

        Args:
            text (str): Raw resume content.

        Returns:
            List[str]: A list of extracted skill strings.
        """
        if not self.model:
            logger.warning("LLM not initialized, cannot extract skills.")
            return ["Error: LLM not initialized"]

        if not text or not text.strip():
            logger.warning("Empty or invalid text provided for skills extraction.")
            return []

        # Use first 2000 characters to capture skills sections while staying within
        # reasonable token limits. Skills are often listed in dedicated sections
        # but may appear throughout the document.
        context = text[:2000]
        prompt = (
            "Extract technical and soft skills from this resume. "
            "Return ONLY a comma-separated list of strings. "
            f"Text: {context}"
        )

        try:
            response = self.model.generate_content(prompt)
            if not response or not response.text:
                logger.warning("Empty response from LLM for skills extraction.")
                return []

            skills = [s.strip() for s in response.text.split(",") if s.strip()]
            if skills:
                logger.info(f"Skills extracted: {len(skills)} items")
                return skills
            else:
                logger.info("No skills found in response.")
                return []

        except ResourceExhausted:
            logger.error("API quota exceeded for skills extraction.")
            return ["Error: API quota exceeded"]
        except ServiceUnavailable:
            logger.error("Gemini service unavailable for skills extraction.")
            return ["Error: Service unavailable"]
        except Exception as e:
            logger.error(f"Skills extraction failed: {e}")
            return [f"Extraction Error: {e}"]
