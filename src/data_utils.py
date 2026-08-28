"""
Dataset utilities: sample-data generation, loading, validation.

Canonical schema used everywhere else in the project
----------------------------------------------------
    id         : unique row id
    user       : handle of the account that authored the post   (network node)
    text       : raw post text                                   (classifier input)
    label      : one of {"normal", "offensive", "hate"}          (ground truth)
    mentions   : comma-separated handles the post refers to      (network edges)
    timestamp  : ISO datetime string                             (optional)

`load_dataset` maps an arbitrary CSV onto this schema so you can plug in a
real research corpus (Davidson, HASOC, HatEval, ...) by only passing the
column names -- no code changes needed.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd

# The three-way scheme this project is built around.
LABELS = ["normal", "offensive", "hate"]

# Colour code reused by the UI so the whole app stays visually consistent.
LABEL_COLORS = {
    "normal":    "#2e9e5b",   # green
    "offensive": "#e0a029",   # amber
    "hate":      "#d64550",   # red
}

# Common ways other datasets spell / encode the same classes.
_LABEL_ALIASES = {
    "0": "hate", "1": "offensive", "2": "normal",          # Davidson et al. numeric scheme
    "hate": "hate", "hateful": "hate", "hate_speech": "hate",
    "offensive": "offensive", "offensive_language": "offensive", "abusive": "offensive",
    "normal": "normal", "neither": "normal", "none": "normal", "clean": "normal",
    "not_offensive": "normal", "non-offensive": "normal",
}


# --------------------------------------------------------------------------- #
#  Sample dataset generation
# --------------------------------------------------------------------------- #
# NOTE ON CONTENT: the "hate" examples below target purely FICTIONAL groups
# (e.g. "greenskins", "moon-dwellers"). This keeps the sample appropriate and
# non-targeting while still giving the classifier the linguistic *pattern* of
# group-directed hostility. The "offensive" class DOES contain real profanity
# (fuck, asshole, bullshit, ...) because an offensive-language classifier must
# learn that vocabulary — it is not directed at any protected group.
# For real accuracy, replace this file with a properly annotated research
# dataset (see README).

_FICTIONAL_GROUPS = [
    "greenskins", "moon-dwellers", "the northfolk", "flatlanders",
    "cloud people", "the outsiders", "sandwalkers", "riverkin",
]

_HATE_TEMPLATES = [
    "all {g} are ruining everything and should be thrown out of here",
    "i hate {g} so much, they don't belong and never will",
    "{g} are worthless parasites dragging this place down",
    "we need to get rid of every single one of the {g}",
    "typical {g}, always causing trouble, the world is better without them",
    "{g} are scum and everyone knows it",
    "why do we even let {g} exist here, they should be silenced",
    "can't stand these {g}, they deserve everything bad coming to them",
    "{g} are a disease and we should wipe them out completely",
    "honestly every {g} makes me sick, they are trash",
    "no place for {g} here, they are animals not people",
    "stop letting {g} in, they are vermin and ruin everything",
]

_OFFENSIVE_TEMPLATES = [
    # Profanity + personal insults (NOT group-directed hate). This vocabulary is
    # what an "offensive language" classifier must recognise, e.g. Davidson et al.
    "fuck off, nobody wants you here",
    "shut the fuck up already, you're insufferable",
    "you're such an asshole, seriously get lost",
    "this is complete bullshit and you know it",
    "piss off you absolute idiot",
    "screw you and your stupid opinion",
    "what the hell is wrong with you, moron",
    "you're a damn moron, just go away",
    "you suck and your posts are total garbage",
    "get lost loser, no one asked for your crap",
    "shut up you pathetic clown, delete your account",
    "you are so freaking stupid it actually hurts",
    "quit whining you crybaby, it's embarrassing",
    "delete your account you brainless troll",
    "what a dumbass take, are you even serious right now",
    "you're an idiot and everyone is laughing at you",
    "damn you're annoying, stop posting your trash",
    "this is the dumbest garbage i've ever read, loser",
]

_NORMAL_TEMPLATES = [
    "had a great coffee this morning, feeling ready for the day",
    "can anyone recommend a good book on machine learning",
    "the match last night was incredible, what a comeback",
    "just finished a lovely hike, the views were stunning",
    "thanks for the birthday wishes everyone, you're the best",
    "really enjoyed the new documentary, highly recommend it",
    "working on my final year project today, wish me luck",
    "made pasta from scratch for the first time, turned out great",
    "looking forward to the weekend, any fun plans",
    "the weather is perfect for a walk in the park today",
    "grateful for my friends and family this week",
    "learning python has been so rewarding, love the community",
    "just adopted a rescue puppy, my heart is so full",
    "the new cafe downtown has amazing pastries, go try it",
    "finished reading a fantastic novel, couldn't put it down",
    "morning run done, feeling energised and ready to code",
    "congrats to the team on shipping the new release today",
    "watching the sunset from the rooftop, absolutely beautiful",
]

# 12 ordinary accounts + 4 "bad actors" who post most of the toxic content
# and get mentioned a lot -> they become the network's top spreaders.
_NORMAL_USERS = [
    "maya_reads", "arjun_dev", "priya_k", "sam_hikes", "leo_music",
    "nina_codes", "raj_photos", "ella_travels", "omar_cooks", "tara_runs",
    "ken_games", "zoe_art",
]
_BAD_ACTORS = ["ragelord", "trollking", "edgy_max", "furygpt"]


def generate_sample_dataset(path: str | None = None,
                            n_rows: int = 300,
                            seed: int = 42) -> pd.DataFrame:
    """Create a deterministic, network-rich sample dataset.

    If `path` is given the CSV is written there; the DataFrame is always
    returned. Bad-actor accounts are wired to be both toxic AND central so
    the spreader ranking has an obvious, demonstrable answer.
    """
    rng = random.Random(seed)
    all_users = _NORMAL_USERS + _BAD_ACTORS
    rows = []
    start = datetime(2026, 1, 1, 9, 0, 0)

    for i in range(n_rows):
        author = _pick_author(rng)
        label = _pick_label(rng, author)
        text = _make_text(rng, label)
        mentions = _pick_mentions(rng, author, all_users)

        # Sprinkle the mentioned handles into the text so text-derived edges
        # also work, and so the data looks like real social posts.
        if mentions:
            text = " ".join(f"@{m}" for m in mentions) + " " + text

        rows.append({
            "id": i + 1,
            "user": author,
            "text": text,
            "label": label,
            "mentions": ",".join(mentions),
            "timestamp": (start + timedelta(minutes=37 * i)).isoformat(),
        })

    df = pd.DataFrame(rows)
    if path:
        df.to_csv(path, index=False)
    return df


def _pick_author(rng: random.Random) -> str:
    # Bad actors are prolific: ~40% of all posts come from just 4 accounts.
    return rng.choice(_BAD_ACTORS) if rng.random() < 0.40 else rng.choice(_NORMAL_USERS)


def _pick_label(rng: random.Random, author: str) -> str:
    r = rng.random()
    if author in _BAD_ACTORS:                       # mostly toxic
        return "hate" if r < 0.60 else ("offensive" if r < 0.90 else "normal")
    return "normal" if r < 0.82 else ("offensive" if r < 0.95 else "hate")  # mostly clean


def _make_text(rng: random.Random, label: str) -> str:
    if label == "hate":
        return rng.choice(_HATE_TEMPLATES).format(g=rng.choice(_FICTIONAL_GROUPS))
    if label == "offensive":
        return rng.choice(_OFFENSIVE_TEMPLATES)
    return rng.choice(_NORMAL_TEMPLATES)


def _pick_mentions(rng: random.Random, author: str, all_users: list[str]) -> list[str]:
    """Choose who this post mentions (these become directed edges author -> mention)."""
    others = [u for u in all_users if u != author]
    if author in _BAD_ACTORS:
        # Bad actors attack many targets -> high out-degree.
        k = rng.randint(1, 3)
        return rng.sample(others, k)
    # Normal users usually mention someone; often amplify a bad actor
    # (reply/quote) -> pushes bad actors' in-degree & PageRank up.
    if rng.random() < 0.30:
        return []
    if rng.random() < 0.45:
        return [rng.choice(_BAD_ACTORS)]
    return [rng.choice(_NORMAL_USERS)]


# --------------------------------------------------------------------------- #
#  Loading / validation for arbitrary CSVs
# --------------------------------------------------------------------------- #
def normalise_label(value) -> str | None:
    """Map any label spelling/encoding onto {normal, offensive, hate}."""
    if value is None:
        return None
    key = str(value).strip().lower()
    return _LABEL_ALIASES.get(key)


def load_dataset(source,
                 text_col: str = "text",
                 label_col: str | None = "label",
                 user_col: str | None = "user",
                 mentions_col: str | None = "mentions") -> pd.DataFrame:
    """Load a CSV (path, file-like, or raw string) into the canonical schema.

    Only `text_col` is strictly required. Missing optional columns are filled
    with sensible defaults so the rest of the app never has to special-case them.
    Returns a DataFrame with columns: id, user, text, label, mentions, timestamp.
    """
    if isinstance(source, str) and "\n" not in source:
        df = pd.read_csv(source)                     # treat as a file path
    elif isinstance(source, str):
        df = pd.read_csv(StringIO(source))           # raw CSV text
    else:
        df = pd.read_csv(source)                     # uploaded file-like object

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)

    if label_col and label_col in df.columns:
        out["label"] = df[label_col].map(normalise_label)
    else:
        out["label"] = None

    out["user"] = df[user_col].astype(str) if (user_col and user_col in df.columns) \
        else [f"user_{i}" for i in range(len(df))]

    if mentions_col and mentions_col in df.columns:
        out["mentions"] = df[mentions_col].fillna("").astype(str)
    else:
        out["mentions"] = ""      # network module can still parse @handles from text

    out["timestamp"] = df["timestamp"] if "timestamp" in df.columns else ""
    out.insert(0, "id", range(1, len(out) + 1))
    # Return in the canonical column order (same as generate_sample_dataset).
    return out[["id", "user", "text", "label", "mentions", "timestamp"]]


def label_distribution(df: pd.DataFrame) -> pd.Series:
    """Counts per class, always in the canonical order, zero-filled."""
    return (df["label"]
            .value_counts()
            .reindex(LABELS)
            .fillna(0)
            .astype(int))
