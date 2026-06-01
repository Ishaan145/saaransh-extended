"""
saaransh-unified.nlp.preprocessing
==================================
Port of the original `universal_preproc.py` and the inline preprocessing
in the original FastAPI `app.py`. Six-stage pipeline as described in
paper §4.1: emoji interpretation, URL / mention / HTML normalisation,
Hinglish detection, conditional translation, domain-noise removal,
tokenisation + stopword / lemma / stem.

The translation stage is left as a hook: in the original repo it calls
the NLLB-200 distilled 600M model. When the weights are not available,
the hook is a no-op and the document passes through unchanged. The rest
of the pipeline is deterministic and runs CPU-only.
"""

from __future__ import annotations
import re
from typing import Callable, Optional

# --------------------------------------------------------------------- maps
EMOJI_MAP = {
    "\U0001F60A": "happy", "\U0001F60D": "love", "\U0001F602": "funny",
    "\U0001F622": "sad",   "\U0001F621": "angry",
    "\U0001F44D": "good",  "\U0001F44E": "bad",
}

SLANG_MAP = {
    "lol": "funny", "imo": "in my opinion", "btw": "by the way",
    "idk": "do not know", "omg": "surprised", "asap": "soon",
    "tbh": "to be honest", "fwiw": "for what it is worth",
}

DOMAIN_NOISE_WORDS = {
    "ref", "hai", "okay", "ok", "hello", "welcome", "slide", "course",
    "video", "module", "session", "training", "instructor", "teacher",
    "points", "unit", "mins", "thank", "thanks", "good", "morning",
    "afternoon", "evening", "lecture", "please", "kindly", "yes", "no",
    "need", "view", "angle", "tbh", "compliance", "change", "saying",
    "great", "super", "forward", "looking", "toh", "thoda",
}

# Trimmed English stopword list (avoids the nltk download dependency).
_STOP_WORDS = {
    "the", "and", "for", "this", "that", "with", "you", "are", "was",
    "were", "will", "shall", "from", "have", "had", "has", "been",
    "but", "not", "can", "may", "would", "should", "could", "a", "an",
    "to", "of", "in", "on", "at", "by", "it", "they", "them", "we",
    "i", "is", "as", "or", "be", "our", "your", "their", "my",
    "about", "into", "than", "then", "so", "if", "when", "do", "does",
    "did", "also", "any", "all", "some", "such", "very", "just", "even",
}

# Hinglish detection keyword list. 15 high-frequency tokens, matches the
# heuristic in the original notebook.
_HINGLISH_KEYWORDS = {
    "hai", "nahi", "kya", "kyun", "matlab", "bhai", "yaar", "achha",
    "bilkul", "thoda", "bahut", "sahi", "galat", "samjha", "karenge",
}

# --------------------------------------------------------------------- regex
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE  = re.compile(r"@\w+")
_HTML_RE     = re.compile(r"<[^>]+>")
_MULTISPACE  = re.compile(r"\s+")
_NONALPHA    = re.compile(r"[^a-zA-Z\u0900-\u097F\s]")  # keep Devanagari


# --------------------------------------------------------------------- stages
def interpret_emojis(text: str) -> str:
    for sym, word in EMOJI_MAP.items():
        text = text.replace(sym, f" {word} ")
    return text


def normalise_urls_html(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    return text


def detect_hinglish(text: str) -> bool:
    """True if at least three Hinglish marker tokens appear (lower-case)."""
    toks = text.lower().split()
    hits = sum(1 for t in toks if t in _HINGLISH_KEYWORDS)
    return hits >= 3


def expand_slang(text: str) -> str:
    out = []
    for tok in text.split():
        out.append(SLANG_MAP.get(tok.lower(), tok))
    return " ".join(out)


def strip_domain_noise(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in DOMAIN_NOISE_WORDS]


def strip_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in _STOP_WORDS]


def simple_lemma(tok: str) -> str:
    # Strip a small set of common suffixes. Cheaper than nltk and avoids
    # the corpus download in a sandbox.
    for suf in ("ing", "ies", "ied", "ed", "es", "s"):
        if len(tok) > 4 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


# --------------------------------------------------------------------- API
def preprocess(
    text: str,
    translator: Optional[Callable[[str], str]] = None,
    keep_devanagari: bool = True,
) -> dict:
    """Run the full six-stage pipeline.

    Returns a dict with cleaned text, the token list, and a small set of
    diagnostics (detected language register, length before / after) so
    that downstream layers can audit the preprocessing.
    """
    original = text
    text = interpret_emojis(text)
    text = normalise_urls_html(text)
    text = expand_slang(text)

    is_hinglish = detect_hinglish(text)
    if is_hinglish and translator is not None:
        text = translator(text)

    if not keep_devanagari:
        text = _NONALPHA.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()

    tokens = text.lower().split()
    tokens = strip_domain_noise(tokens)
    tokens = strip_stopwords(tokens)
    tokens = [simple_lemma(t) for t in tokens]
    cleaned = " ".join(tokens)

    return {
        "original": original,
        "cleaned": cleaned,
        "tokens": tokens,
        "is_hinglish": is_hinglish,
        "len_before": len(original),
        "len_after": len(cleaned),
    }
