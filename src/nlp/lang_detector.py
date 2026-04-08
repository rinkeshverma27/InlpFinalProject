"""
src/nlp/lang_detector.py — Lightweight language classifier using lingua-py.

Falls back to a simple Unicode script heuristic if lingua is not installed.
Cost: < 1ms per headline.
"""

from typing import List, Literal
from src.utils.logger import get_logger

log = get_logger("lang_detector")

LangLabel = Literal["en", "hi", "hinglish", "unknown"]

# ── Try to load lingua ────────────────────────────────────────────────────────
try:
    from lingua import Language, LanguageDetectorBuilder
    _DETECTOR = (
        LanguageDetectorBuilder
        .from_languages(Language.ENGLISH, Language.HINDI)
        .with_minimum_relative_distance(0.0)
        .build()
    )
    _USE_LINGUA = True
    log.info("Language detector: using lingua-py.")
except ImportError:
    _DETECTOR    = None
    _USE_LINGUA  = False
    log.warning(
        "lingua not installed — falling back to Unicode heuristic.\n"
        "Install: pip install lingua-language-detector"
    )


# ── Unicode heuristic fallback ────────────────────────────────────────────────
def _devanagari_ratio(text: str) -> float:
    deva = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return deva / max(len(text), 1)


def _heuristic_detect(text: str, threshold: float) -> LangLabel:
    d_ratio = _devanagari_ratio(text)
    if d_ratio >= threshold:
        return "hi"
    if d_ratio > 0.05:    # mixed — Devanagari chars but also Latin
        return "hinglish"
    return "en"


# ── Public API ────────────────────────────────────────────────────────────────

def detect_language(text: str, confidence_threshold: float = 0.85) -> LangLabel:
    """
    Detect language of a single text.

    Returns:
        "en"       — English
        "hi"       — Pure Hindi (Devanagari)
        "hinglish" — Code-switched Hindi+English
        "unknown"  — Below confidence threshold → dropped at train time
    """
    if not text or not text.strip():
        return "unknown"

    devanagari_ratio = _devanagari_ratio(text)

    # Hinglish heuristic: has some Devanagari + significant Latin
    if 0.05 < devanagari_ratio < 0.85:
        return "hinglish"

    if not _USE_LINGUA:
        return _heuristic_detect(text, confidence_threshold)

    result = _DETECTOR.detect_language_of(text)
    if result is None:
        return "unknown"

    # lingua confidence
    try:
        conf_values = _DETECTOR.compute_language_confidence_values(text)
        top_conf    = conf_values[0].value if conf_values else 0.0
    except Exception:
        top_conf = 1.0   # if confidence API fails, trust the label

    if top_conf < confidence_threshold:
        return "unknown"

    from lingua import Language
    if result == Language.HINDI:
        return "hi"
    if result == Language.ENGLISH:
        return "en"
    return "unknown"


def detect_batch(texts: List[str], confidence_threshold: float = 0.85) -> List[LangLabel]:
    """Batch language detection. Returns list of labels in same order as input."""
    return [detect_language(t, confidence_threshold) for t in texts]
