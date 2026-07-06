"""
Multilingual support utilities: language detection and translation.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "hi": "Hindi",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "ru": "Russian",
}


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of a text string.
    Returns (lang_code, confidence).
    """
    try:
        from langdetect import detect, detect_langs
        langs = detect_langs(text)
        if langs:
            top = langs[0]
            return top.lang, round(top.prob, 3)
        return "en", 0.0
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en", 0.0


def translate_text(
    text: str,
    target_lang: str = "en",
    source_lang: str = "auto",
) -> str:
    """
    Translate text to target language using googletrans.
    Falls back to original text on failure.
    """
    if not text.strip():
        return text
    try:
        from googletrans import Translator
        translator = Translator()
        src = "auto" if source_lang == "auto" else source_lang
        result = translator.translate(text, dest=target_lang, src=src)
        return result.text
    except Exception as e:
        logger.warning(f"Translation failed ({source_lang} -> {target_lang}): {e}")
        return text


class MultilingualHandler:
    """
    Handles multilingual queries:
    1. Detect query language.
    2. Translate query to English for retrieval.
    3. Translate answer back to original language.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def process_query(self, query: str) -> Tuple[str, str, float]:
        """
        Returns (english_query, detected_lang, confidence).
        """
        if not self.enabled:
            return query, "en", 1.0

        lang, confidence = detect_language(query)
        if lang == "en" or confidence < 0.5:
            return query, "en", confidence

        english_query = translate_text(query, target_lang="en", source_lang=lang)
        logger.info(f"Query translated: {lang} -> en (confidence={confidence})")
        return english_query, lang, confidence

    def process_answer(self, answer: str, target_lang: str) -> str:
        """
        Translate the answer back to the user's original language.
        """
        if not self.enabled or target_lang == "en":
            return answer
        translated = translate_text(answer, target_lang=target_lang, source_lang="en")
        logger.info(f"Answer translated: en -> {target_lang}")
        return translated

    @staticmethod
    def language_name(code: str) -> str:
        return LANGUAGE_NAMES.get(code, code.upper())