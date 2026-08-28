# 🕸️ Online Hate Speech Network Analyser

**Classify online messages as `Normal` / `Offensive` / `Hate`, then analyse the social network to reveal *who spreads the hate* — in one connected pipeline.**

> **Project novelty.** Existing tools typically do *either* hate-speech
> classification *or* social-network analysis. This project **chains them
> together**: the classifier's output feeds a network layer, so you move from
> *“is this message hateful?”* to *“which accounts are driving the hate, and
> how far does their reach extend?”* — all in a single, reproducible workflow.

---

## ✨ What it does

| Stage | What happens | Tech |
|-------|--------------|------|
| **1. Classify** | Every post is labelled Normal / Offensive / Hate with a confidence score. Shows accuracy, macro-F1, a confusion matrix and the top indicative words per class. | TF-IDF + Logistic Regression (scikit-learn) |
| **2. Connect** | Posts are turned into a **directed graph**: an edge goes from an author to each account they mention, weighted by how toxic the post was. | NetworkX |
| **3. Rank spreaders** | Each user gets a **spreader score = network influence × content toxicity**, exposing the accounts that are both toxic *and* central. | PageRank / betweenness / degree centrality |

Everything is wrapped in a 5-tab Streamlit app: **Overview → Data → Train & Evaluate → Classify → Network & Spreaders**.

---

## 🚀 Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

Then in the browser: open **📁 Data → “Load sample dataset”**, go to **🎯 Train & Evaluate → “Train model”**, and finally **🕸️ Network & Spreaders → “Build network & rank spreaders”**.

---

## 📁 Project structure

```
online-hate-speech-network-analyser/
├── app.py                 # Streamlit UI (5 tabs) — thin presentation layer
├── requirements.txt
├── README.md
├── data/
│   └── sample_hate_speech.csv    # synthetic, network-rich sample dataset
└── src/
    ├── preprocessing.py   # regex text cleaning + @mention extraction
    ├── classifier.py      # HateSpeechClassifier: train / predict / save / load
    ├── network.py         # graph building, centrality, spreader ranking, Plotly viz
    └── data_utils.py      # schema, loading/validation, sample-data generator
```

The UI never contains analysis logic — it only calls into `src/`. That means
you can **restyle or rebuild the interface freely** (which is the plan) without
touching the models or the network maths.

---

## 📊 Using your own dataset

Upload any CSV in the **Data** tab. The only required field is a **text**
column; everything else is optional and mapped through dropdowns.

| Canonical field | Meaning | Required? |
|-----------------|---------|-----------|
| `text` | the post / tweet / comment | ✅ yes |
| `label` | `hate` / `offensive` / `normal` (many spellings & the Davidson `0/1/2` scheme are auto-recognised) | for training |
| `user` | author handle → becomes a network **node** | for network |
| `mentions` | comma-separated handles → become network **edges** (if absent, `@handles` are parsed from the text) | optional |

Great public corpora to plug in: **Davidson et al. (2017)**, **HASOC**,
**HatEval**, **OLID / OffensEval**.

---

## 🧮 How the spreader score works (for your report / viva)

A hate-speech **spreader** must be *both* toxic *and* influential:

```
influence = 0.5·PageRank + 0.3·degree + 0.2·betweenness   (each min-max normalised → 0..1)
toxicity  = ( 1.0·#hate + 0.5·#offensive ) / #posts        (0..1)
score     = influence^w · toxicity^(1−w)                    (weighted geometric mean → rescaled 0..100)
```

* `w` (the **Reach ↔ Content** slider) trades off network reach against content toxicity; `w = 0.5` is balanced.
* A viral but clean news account → `toxicity ≈ 0` → score ≈ 0 (correctly *not* a spreader).
* A nasty account nobody listens to → `influence ≈ 0` → score ≈ 0.
* Only accounts that are **toxic AND central** rise to the top.

This is the defensible, explainable core of the “network + hate speech” novelty.

---

## 🛣️ Roadmap — where to extend next

The code marks natural extension points with `# EXTEND:` comments.

- **Stronger models** — swap the TF-IDF pipeline for a fine-tuned transformer
  (e.g. `bert-base`, `HateBERT`, `distil-roBERTa`); `classifier.py` already
  isolates the model behind one class.
- **Live ingestion** — replace the CSV loader with a Twitter/X, Reddit, or
  Mastodon API pull to analyse real conversations.
- **Temporal spread** — the dataset carries `timestamp`; add an animation of
  how hate propagates through the network over time.
- **Community detection** — run Louvain / Leiden to find the *clusters* that
  amplify a spreader, not just individual accounts.
- **Explainability** — add SHAP/LIME to justify each classification.
- **Cascade modelling** — model information diffusion (Independent Cascade /
  Linear Threshold) to *predict* which accounts a piece of hate will reach.

---

## ⚠️ Notes on the sample data & ethics

The bundled sample dataset is **synthetic**. Its “hate” examples deliberately
target **fictional groups** (e.g. *greenskins*, *moon-dwellers*) so the file is
appropriate to ship while still giving the classifier the linguistic *pattern*
of group-directed hostility. Numbers on this tiny set are illustrative — use a
real annotated corpus for meaningful accuracy.

This tool is intended for **research, moderation-support, and academic study**.
Automated hate-speech systems can carry bias (e.g. over-flagging dialects);
treat outputs as decision-support, keep a human in the loop, and validate on
representative data before drawing conclusions about real people.

---

*Built as an extensible foundation — `src/` holds the logic, `app.py` holds the UI.*
