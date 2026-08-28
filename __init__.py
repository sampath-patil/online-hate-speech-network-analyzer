"""
Online Hate Speech Network Analyser
-----------------------------------
Core package containing the four building blocks of the pipeline:

    preprocessing  -> text cleaning / normalisation
    classifier     -> TF-IDF + Logistic Regression (Hate / Offensive / Normal)
    network        -> user-mention graph, centrality, spreader ranking
    data_utils     -> loading, validation, sample-dataset generation

The Streamlit UI (app.py) is only a thin presentation layer on top of
these modules, so you can reuse / replace any single piece without
touching the rest.
"""

__version__ = "0.1.0"
