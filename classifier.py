"""
Hate-speech classifier: TF-IDF features + Logistic Regression.

Why this model?
    * Fast to train (seconds), runs on a laptop, no GPU.
    * `predict_proba` gives calibrated-ish confidence for the UI.
    * `class_weight="balanced"` copes with the usual heavy class imbalance
      (real hate-speech corpora are mostly "offensive"/"normal").
    * Coefficients are inspectable -> `top_features()` shows *why* the model
      flags something, which is great for a project report / viva.

Swap-in points for later work are marked with  # EXTEND:
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from .preprocessing import clean_text
from .data_utils import LABELS


class HateSpeechClassifier:
    """Thin, save-able wrapper around a scikit-learn text pipeline."""

    def __init__(self,
                 model_type: str = "logreg",
                 ngram_range: tuple[int, int] = (1, 2),
                 max_features: int = 20000,
                 min_df: int = 1,
                 C: float = 1.0):
        self.model_type = model_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.C = C
        self.pipeline: Pipeline | None = None
        self.is_trained = False
        self.metrics: dict | None = None

    # ------------------------------------------------------------------ #
    #  Build & train
    # ------------------------------------------------------------------ #
    def _build_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            preprocessor=clean_text,        # our regex cleaner runs per document
            stop_words="english",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
        )

        if self.model_type == "svm":
            # LinearSVC has no predict_proba -> wrap it so the UI still gets scores.
            base = LinearSVC(class_weight="balanced", C=self.C)
            clf = CalibratedClassifierCV(base, cv=3)
        else:  # EXTEND: add "transformer" here to plug in a fine-tuned BERT, etc.
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=self.C,
                n_jobs=-1,
            )

        return Pipeline([("tfidf", vectorizer), ("clf", clf)])

    def train(self,
              texts: list[str],
              labels: list[str],
              test_size: float = 0.25,
              seed: int = 42) -> dict:
        """Fit the pipeline and return a metrics dict for the UI / report.

        Returns keys: accuracy, macro_f1, weighted_f1, report (per-class dict),
        confusion_matrix (list-of-lists in LABELS order), labels, n_train, n_test.
        """
        texts = list(texts)
        labels = list(labels)

        # Stratify only when every class has at least 2 samples (else sklearn errors).
        counts = pd.Series(labels).value_counts()
        stratify = labels if counts.min() >= 2 and len(counts) > 1 else None

        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=test_size, random_state=seed, stratify=stratify
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_tr, y_tr)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_te)
        present = [lbl for lbl in LABELS if lbl in set(y_te) | set(y_pred)]

        self.metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            "report": classification_report(
                y_te, y_pred, labels=present, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_te, y_pred, labels=present).tolist(),
            "labels": present,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }
        return self.metrics

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def _check(self):
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Classifier is not trained yet. Call train() first.")

    def predict(self, texts) -> list[str]:
        self._check()
        if isinstance(texts, str):
            texts = [texts]
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts) -> pd.DataFrame:
        """Return a DataFrame of class probabilities (columns = class names)."""
        self._check()
        if isinstance(texts, str):
            texts = [texts]
        proba = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.named_steps["clf"].classes_)
        return pd.DataFrame(proba, columns=classes)

    def predict_df(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Add `predicted_label` and `confidence` columns to a copy of `df`."""
        self._check()
        out = df.copy()
        proba = self.predict_proba(out[text_col].tolist())
        out["predicted_label"] = proba.idxmax(axis=1).values
        out["confidence"] = proba.max(axis=1).values
        return out

    # ------------------------------------------------------------------ #
    #  Interpretability & persistence
    # ------------------------------------------------------------------ #
    def top_features(self, n: int = 15) -> dict[str, list[str]]:
        """Most indicative words/bigrams per class (LogisticRegression only).

        Handy for the report: shows the vocabulary the model associates with
        hate vs. offensive vs. normal.
        """
        self._check()
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return {}
        vocab = np.array(self.pipeline.named_steps["tfidf"].get_feature_names_out())
        result = {}
        for idx, cls in enumerate(clf.classes_):
            # Binary case: coef_ has a single row; positive = this class.
            row = clf.coef_[idx] if clf.coef_.shape[0] > 1 else clf.coef_[0]
            top = np.argsort(row)[-n:][::-1]
            result[cls] = vocab[top].tolist()
        return result

    def save(self, path: str):
        self._check()
        joblib.dump({"pipeline": self.pipeline, "metrics": self.metrics,
                     "model_type": self.model_type}, path)

    @classmethod
    def load(cls, path: str) -> "HateSpeechClassifier":
        blob = joblib.load(path)
        obj = cls(model_type=blob.get("model_type", "logreg"))
        obj.pipeline = blob["pipeline"]
        obj.metrics = blob.get("metrics")
        obj.is_trained = True
        return obj
