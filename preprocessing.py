"""
Text pre-processing for hate-speech classification.

Kept deliberately dependency-light (pure Python + regex) so the project
installs and runs anywhere without downloading NLTK/spaCy corpora.
If you later want lemmatisation or stop-word lists from NLTK, swap the
`clean_text` body -- the rest of the pipeline calls this one function.
"""

import re

# Pre-compiled regexes (compiled once, reused for every row -> fast on big files)
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE  = re.compile(r"@\w+")
_HASHTAG_RE  = re.compile(r"#(\w+)")          # keep the word, drop the '#'
_NON_ALPHA   = re.compile(r"[^a-z\s']")       # keep letters, spaces, apostrophes
_MULTISPACE  = re.compile(r"\s+")


def extract_mentions(text: str) -> list[str]:
    """Return the list of @handles mentioned in a piece of text.

    Used by the network module when a dataset does not ship an explicit
    `mentions` column -- we can then reconstruct edges straight from the text.
    """
    if not isinstance(text, str):
        return []
    return [m.lower().lstrip("@") for m in _MENTION_RE.findall(text)]


def clean_text(text: str,
               remove_mentions: bool = True,
               keep_hashtag_word: bool = True) -> str:
    """Normalise a raw social-media string into model-ready tokens.

    Steps: lower-case -> strip URLs -> handle @mentions and #hashtags ->
    drop non-alphabetic noise -> collapse whitespace.

    Parameters
    ----------
    text : str
        Raw input string (a tweet / post / comment).
    remove_mentions : bool
        If True, @handles are deleted (they rarely carry hate signal and
        would otherwise leak user identity into the text features).
    keep_hashtag_word : bool
        If True, "#MondayMotivation" -> "mondaymotivation" (word kept);
        if False the whole hashtag token is dropped.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)

    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)

    if keep_hashtag_word:
        text = _HASHTAG_RE.sub(r"\1", text)
    else:
        text = _HASHTAG_RE.sub(" ", text)

    text = _NON_ALPHA.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def clean_series(texts) -> list[str]:
    """Vectorised convenience wrapper: clean an iterable of strings."""
    return [clean_text(t) for t in texts]
