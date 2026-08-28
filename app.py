"""
Online Hate Speech Network Analyser  --  Streamlit UI
=====================================================

Run with:   streamlit run app.py

This file is only the *presentation* layer. All the logic lives in src/:
    src/preprocessing.py  src/classifier.py  src/network.py  src/data_utils.py

The five tabs walk through the full pipeline end-to-end:
    Overview -> Data -> Train & Evaluate -> Classify -> Network & Spreaders

UI note: status indicators and badges are drawn with plain CSS (coloured dots /
pills) instead of emoji, so they render identically on every OS. Each tab is its
own render_* function and shared state lives in st.session_state, so you can
restyle freely without touching the analysis.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `src` importable no matter where streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import (LABELS, LABEL_COLORS, generate_sample_dataset,
                            load_dataset, label_distribution)
from src.classifier import HateSpeechClassifier
from src import network as net

# --------------------------------------------------------------------------- #
#  Page config + theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Hate Speech Network Analyser",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .block-container { padding-top: 1.4rem; }
      .hero {
        background: linear-gradient(120deg, #1f2a44 0%, #3b2f63 55%, #6d3b6b 100%);
        color: #fff; padding: 26px 30px; border-radius: 16px; margin-bottom: 8px;
      }
      .hero h1 { margin: 0; font-size: 1.9rem; letter-spacing: .3px; }
      .hero p  { margin: 6px 0 0; opacity: .92; font-size: 1.02rem; }
      .pill {
        display:inline-block; padding:3px 11px; border-radius:20px;
        font-size:.72rem; font-weight:700; letter-spacing:.4px;
        background:rgba(255,255,255,.16); margin-right:6px;
      }
      .badge {
        display:inline-block; padding:4px 14px; border-radius:14px;
        color:#fff; font-weight:700; font-size:1rem; letter-spacing:.5px;
      }
      /* sidebar logo */
      .logo { display:flex; align-items:center; gap:10px;
              font-weight:800; font-size:1.2rem; }
      .logo-mark { width:18px; height:18px; border-radius:5px;
                   background:linear-gradient(135deg,#3b2f63,#8a4a86); }
      /* pipeline status rows */
      .status-row { display:flex; align-items:center; gap:9px; margin:7px 0;
                    font-size:.94rem; }
      .status-step { opacity:.45; width:14px; text-align:center; }
      .status-name { flex:1; }
      .status-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
      .status-text { font-size:.8rem; font-weight:700; }
      /* tab bar: sizing/weight here; colours are set per-theme in inject_theme() */
      .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid; }
      .stTabs [data-baseweb="tab"] { padding: 8px 18px; }
      .stTabs [data-baseweb="tab"] * { font-size: 1.06rem !important; font-weight: 700 !important; }
      .stTabs [aria-selected="true"] { border-bottom: 3px solid; }
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#888")
    return f'<span class="badge" style="background:{color}">{label.upper()}</span>'


def status_row(step: str, name: str, ok: bool, on_text: str, off_text: str) -> str:
    color = "#2e9e5b" if ok else "#b9bdc9"
    text = on_text if ok else off_text
    return (f'<div class="status-row">'
            f'<span class="status-step">{step}</span>'
            f'<span class="status-name">{name}</span>'
            f'<span class="status-dot" style="background:{color}"></span>'
            f'<span class="status-text" style="color:{color}">{text}</span></div>')


# --------------------------------------------------------------------------- #
#  Theme (runtime dark / light toggle)
# --------------------------------------------------------------------------- #
# Streamlit's .streamlit/config.toml theme is fixed at startup and can't switch
# live, so the light/dark toggle is done by injecting a CSS "skin" on every run
# based on st.session_state["dark_mode"]. Self-coloured components (hero, badge,
# pill) are re-asserted so the broad text override never washes them out.
LIGHT_THEME = {
    "bg": "#ffffff", "panel": "#f4f4f8", "text": "#1f2333", "muted": "#5b6070",
    "border": "#e6e6ef", "card": "#fbfbfe", "input": "#ffffff", "accent": "#6d3b6b",
}
DARK_THEME = {
    "bg": "#10131c", "panel": "#171b27", "text": "#e7e9f2", "muted": "#9aa0b4",
    "border": "#2b3040", "card": "#161a26", "input": "#1e2331", "accent": "#c78ac2",
}


def active_theme() -> dict:
    return DARK_THEME if st.session_state.get("dark_mode", False) else LIGHT_THEME


def inject_theme():
    """Emit the CSS skin for the currently-selected theme."""
    c = active_theme()
    st.markdown(
        f"""
        <style>
          /* page + sidebar surfaces */
          .stApp {{ background: {c['bg']} !important; }}
          [data-testid="stHeader"] {{ background: transparent !important; }}
          section[data-testid="stSidebar"] {{ background: {c['panel']} !important; }}
          /* broad text colour (NOT `span`, so inline-coloured badges survive) */
          .stApp, .stApp p, .stApp li, .stApp label,
          .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
          [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
          [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: {c['text']} !important;
          }}
          /* re-assert self-coloured components (placed AFTER the broad rule) */
          .hero, .hero p, .hero h1, .hero b, .hero div, .hero span {{ color: #ffffff !important; }}
          .badge {{ color: #ffffff !important; }}
          .pill  {{ color: #ffffff !important; }}
          .status-name {{ color: {c['text']} !important; }}
          /* cards / info & warning boxes pick up the surface colour */
          [data-testid="stNotification"], .stAlert {{
            background: {c['card']} !important; border: 1px solid {c['border']} !important;
          }}
          /* tab colours (structure/size stay in the static block) */
          .stTabs [data-baseweb="tab-list"] {{ border-bottom-color: {c['border']} !important; }}
          .stTabs [data-baseweb="tab"] * {{ color: {c['text']} !important; }}
          .stTabs [data-baseweb="tab"]:hover * {{ color: {c['accent']} !important; }}
          .stTabs [aria-selected="true"] {{ border-bottom-color: {c['accent']} !important; }}
          .stTabs [aria-selected="true"] * {{ color: {c['accent']} !important; }}
          /* inputs / textareas / selects */
          .stTextArea textarea, .stTextInput input,
          [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
            background: {c['input']} !important; color: {c['text']} !important;
          }}
          textarea::placeholder {{ color: {c['muted']} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Session state
# --------------------------------------------------------------------------- #
for key, default in {
    "df": None, "clf": None, "pred_df": None,
    "ranking": None, "graph": None,
}.items():
    st.session_state.setdefault(key, default)


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="logo"><span class="logo-mark"></span>Analyser</div>',
                    unsafe_allow_html=True)
        st.caption("Hate-speech classification + spreader-network analysis in one pipeline.")

        st.toggle("Dark mode", key="dark_mode",
                  help="Switch the whole app between light and dark themes.")
        st.divider()

        df = st.session_state.df
        clf = st.session_state.clf
        st.markdown("**Pipeline status**")
        html = (
            status_row("1", "Data", df is not None, "loaded", "not loaded") +
            status_row("2", "Model", bool(clf and clf.is_trained), "trained", "not trained") +
            status_row("3", "Network", st.session_state.ranking is not None, "built", "not built")
        )
        st.markdown(html, unsafe_allow_html=True)

        if df is not None:
            st.divider()
            st.metric("Rows loaded", len(df))
            st.metric("Labelled rows", int(df["label"].notna().sum()))

        st.divider()
        st.caption("Extensible base — logic lives in src/, UI in app.py.")


# --------------------------------------------------------------------------- #
#  Tab 1 — Overview
# --------------------------------------------------------------------------- #
def render_overview():
    st.markdown(
        """
        <div class="hero">
          <h1>Online Hate Speech Network Analyser</h1>
          <p>Detect whether a message is <b>Normal</b>, <b>Offensive</b> or <b>Hate</b> —
          then map the social network to reveal <b>who spreads it</b>.</p>
          <div style="margin-top:12px">
            <span class="pill">TEXT CLASSIFICATION</span>
            <span class="pill">NETWORK ANALYSIS</span>
            <span class="pill">SPREADER RANKING</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1 · Classify")
        st.markdown(
            "A **TF-IDF + Logistic Regression** model labels every post as "
            "Normal / Offensive / Hate, with confidence scores and a confusion matrix."
        )
    with c2:
        st.markdown("#### 2 · Connect")
        st.markdown(
            "Posts become a **directed graph** — an edge runs from an author to "
            "each account they mention, weighted by how toxic those posts are."
        )
    with c3:
        st.markdown("#### 3 · Rank spreaders")
        st.markdown(
            "We combine **network influence × content toxicity** into a single "
            "*spreader score*, exposing the accounts driving the hate."
        )

    st.divider()
    st.markdown("#### The novelty")
    st.info(
        "Most tools do **either** hate-speech detection **or** social-network analysis. "
        "This project chains them: the classifier's output feeds the network layer, so you "
        "move from *“is this hateful?”* to *“who is spreading it, and how far does it reach?”* — "
        "in a single, reproducible pipeline."
    )

    st.markdown("#### How to use this app")
    st.markdown(
        "1. Open the **Data** tab → *Load sample dataset* (or upload your own CSV).\n"
        "2. Go to **Train & Evaluate** → train the classifier and read the metrics.\n"
        "3. Try **Classify** on any sentence you type.\n"
        "4. Open **Network & Spreaders** to see the graph and the ranked spreaders."
    )
    st.caption("Everything runs locally. No data leaves your machine.")


# --------------------------------------------------------------------------- #
#  Tab 2 — Data
# --------------------------------------------------------------------------- #
def render_data():
    st.subheader("Data upload & exploration")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Option A — use the built-in sample**")
        st.caption("A synthetic, network-rich dataset (fictional targets) so you can try "
                   "the whole pipeline instantly.")
        if st.button("Load sample dataset", type="primary", use_container_width=True):
            st.session_state.df = generate_sample_dataset()
            st.session_state.clf = None
            st.session_state.pred_df = None
            st.session_state.ranking = None
            st.success("Sample dataset loaded.")

    with right:
        st.markdown("**Option B — upload your own CSV**")
        up = st.file_uploader("CSV with at least a text column", type=["csv"])
        if up is not None:
            preview = pd.read_csv(up)
            up.seek(0)
            cols = list(preview.columns)
            st.caption("Map your columns onto the schema:")
            cc1, cc2 = st.columns(2)
            text_col = cc1.selectbox("Text column", cols, index=_guess(cols, ["text", "tweet", "content", "message"]))
            label_col = cc2.selectbox("Label column (optional)", ["<none>"] + cols,
                                      index=_guess(cols, ["label", "class", "category"], offset=1))
            user_col = cc1.selectbox("User column (optional)", ["<none>"] + cols,
                                     index=_guess(cols, ["user", "username", "author", "handle"], offset=1))
            ment_col = cc2.selectbox("Mentions column (optional)", ["<none>"] + cols,
                                     index=_guess(cols, ["mentions", "reply_to", "target"], offset=1))
            if st.button("Load uploaded CSV", type="primary"):
                st.session_state.df = load_dataset(
                    up, text_col=text_col,
                    label_col=None if label_col == "<none>" else label_col,
                    user_col=None if user_col == "<none>" else user_col,
                    mentions_col=None if ment_col == "<none>" else ment_col,
                )
                st.session_state.clf = None
                st.session_state.pred_df = None
                st.session_state.ranking = None
                st.success("Uploaded dataset loaded.")

    df = st.session_state.df
    if df is None:
        st.info("Load the sample dataset or upload a CSV to begin.")
        return

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posts", len(df))
    m2.metric("Unique users", df["user"].nunique())
    m3.metric("Labelled", int(df["label"].notna().sum()))
    m4.metric("With mentions", int((df["mentions"].astype(str).str.len() > 0).sum()))

    cA, cB = st.columns([1.1, 1])
    with cA:
        st.markdown("**Sample of the data**")
        st.dataframe(df.head(15), use_container_width=True, height=360)
    with cB:
        st.markdown("**Label distribution**")
        if df["label"].notna().any():
            dist = label_distribution(df)
            fig = px.bar(x=dist.index, y=dist.values,
                         color=dist.index, color_discrete_map=LABEL_COLORS,
                         labels={"x": "", "y": "posts"})
            fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No labels found — you can still classify & build the network, "
                       "but you won't be able to train/evaluate against ground truth.")


def _guess(cols, candidates, offset=0):
    lc = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(lc):
            if cand == c or cand in c:
                return i + offset
    return 0


# --------------------------------------------------------------------------- #
#  Tab 3 — Train & Evaluate
# --------------------------------------------------------------------------- #
def render_train():
    st.subheader("Train & evaluate the classifier")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset in the **Data** tab first.")
        return
    if not df["label"].notna().any():
        st.warning("This dataset has no labels, so the model can't be trained. "
                   "Upload a labelled CSV or use the sample dataset.")
        return

    train_df = df.dropna(subset=["label"])
    c1, c2, c3, c4 = st.columns(4)
    model_type = c1.selectbox("Model", ["logreg", "svm"],
                              format_func=lambda m: "Logistic Regression" if m == "logreg" else "Linear SVM")
    ngram = c2.selectbox("N-grams", ["1", "1-2", "1-3"], index=1)
    test_size = c3.slider("Test split", 0.1, 0.4, 0.25, 0.05)
    C = c4.select_slider("Regularisation C", [0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)

    if st.button("Train model", type="primary"):
        ngram_range = {"1": (1, 1), "1-2": (1, 2), "1-3": (1, 3)}[ngram]
        with st.spinner("Training…"):
            clf = HateSpeechClassifier(model_type=model_type, ngram_range=ngram_range, C=C)
            clf.train(train_df["text"].tolist(), train_df["label"].tolist(), test_size=test_size)
            st.session_state.clf = clf
            # Pre-compute predictions for the whole dataset (used by other tabs).
            st.session_state.pred_df = clf.predict_df(df)
            st.session_state.ranking = None
        st.success("Model trained.")

    clf = st.session_state.clf
    if not (clf and clf.is_trained):
        st.info("Set your options and click **Train model**.")
        return

    m = clf.metrics
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    k2.metric("Macro F1", f"{m['macro_f1']:.3f}")
    k3.metric("Weighted F1", f"{m['weighted_f1']:.3f}")
    k4.metric("Train / Test", f"{m['n_train']} / {m['n_test']}")

    st.caption("On a small sample dataset these numbers are indicative — the pipeline, "
               "metrics and workflow are what matter. Plug in a larger corpus for real scores.")

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.markdown("**Confusion matrix**")
        lbls = m["labels"]
        cm = m["confusion_matrix"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=[f"pred {l}" for l in lbls], y=[f"true {l}" for l in lbls],
            text=cm, texttemplate="%{text}", colorscale="Purples", showscale=False))
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.markdown("**Per-class performance**")
        report = m["report"]
        rows = [{"class": c,
                 "precision": round(report[c]["precision"], 3),
                 "recall": round(report[c]["recall"], 3),
                 "f1": round(report[c]["f1-score"], 3),
                 "support": int(report[c]["support"])}
                for c in m["labels"] if c in report]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**What words drive each class?** (top indicative terms)")
    feats = clf.top_features(n=12)
    if feats:
        fcols = st.columns(len(feats))
        for col, (cls, words) in zip(fcols, feats.items()):
            col.markdown(badge(cls), unsafe_allow_html=True)
            col.write(", ".join(words))
    else:
        st.caption("Top-feature view is available for the Logistic Regression model.")


# --------------------------------------------------------------------------- #
#  Tab 4 — Classify
# --------------------------------------------------------------------------- #
def render_classify():
    st.subheader("Classify text")
    clf = st.session_state.clf
    if not (clf and clf.is_trained):
        st.info("Train a model in the **Train & Evaluate** tab first.")
        return

    st.markdown("**Single message**")
    txt = st.text_area("Type or paste a message", "you don't belong here and never will",
                       height=90)
    if st.button("Classify", type="primary"):
        proba = clf.predict_proba([txt]).iloc[0]
        pred = proba.idxmax()
        st.markdown(f"Prediction: {badge(pred)} &nbsp; "
                    f"<span style='color:#666'>confidence {proba.max():.0%}</span>",
                    unsafe_allow_html=True)
        pf = proba.reindex(LABELS).fillna(0).reset_index()
        pf.columns = ["class", "probability"]
        fig = px.bar(pf, x="probability", y="class", orientation="h",
                     color="class", color_discrete_map=LABEL_COLORS, range_x=[0, 1])
        fig.update_layout(showlegend=False, height=220, margin=dict(t=6, b=6))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Batch — one message per line**")
    batch = st.text_area("Messages", height=120,
                         placeholder="great game last night\nyou are such an idiot\n…")
    if st.button("Classify batch"):
        lines = [l for l in batch.splitlines() if l.strip()]
        if lines:
            proba = clf.predict_proba(lines)
            out = pd.DataFrame({"text": lines,
                                "prediction": proba.idxmax(axis=1).values,
                                "confidence": proba.max(axis=1).round(3).values})
            st.dataframe(out, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
#  Tab 5 — Network & Spreaders
# --------------------------------------------------------------------------- #
def render_network():
    st.subheader("Network & spreader analysis")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset in the **Data** tab first.")
        return

    has_pred = st.session_state.pred_df is not None
    has_truth = df["label"].notna().any()

    c1, c2, c3 = st.columns([1.2, 1, 1])
    source_opts = []
    if has_truth:
        source_opts.append("Ground-truth labels")
    if has_pred:
        source_opts.append("Model predictions")
    if not source_opts:
        st.warning("No labels and no trained model. Train a model or load labelled data.")
        return
    source = c1.selectbox("Label source for toxicity", source_opts)
    influence_weight = c2.slider("Reach ↔ Content", 0.0, 1.0, 0.5, 0.1,
                                 help="1.0 = pure network reach, 0.0 = pure content toxicity")
    top_k = c3.slider("Highlight top-K", 3, 10, 5)

    # Choose which dataframe / column feeds the network.
    if source == "Model predictions":
        work_df = st.session_state.pred_df
        label_col = "predicted_label"
    else:
        work_df = df.dropna(subset=["label"])
        label_col = "label"

    if st.button("Build network & rank spreaders", type="primary"):
        with st.spinner("Analysing network…"):
            ranking, G = net.spreader_ranking(work_df, label_col=label_col,
                                               influence_weight=influence_weight)
            st.session_state.ranking = ranking
            st.session_state.graph = G

    ranking = st.session_state.ranking
    G = st.session_state.graph
    if ranking is None or G is None:
        st.info("Set your options and click **Build network & rank spreaders**.")
        return

    s = net.graph_summary(G)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accounts (nodes)", s["nodes"])
    k2.metric("Interactions (edges)", s["edges"])
    k3.metric("Density", s["density"])
    k4.metric("Communities", s["weak_components"])

    gcol, tcol = st.columns([1.35, 1])
    with gcol:
        st.markdown("**Interaction network** — redder & bigger = bigger spreader")
        st.plotly_chart(net.build_plotly_figure(G, ranking, top_k=top_k),
                        use_container_width=True)
    with tcol:
        st.markdown(f"**Top {top_k} hate-speech spreaders**")
        top = ranking.head(top_k)
        fig = px.bar(top, x="spreader_score", y="user", orientation="h",
                     color="spreader_score", color_continuous_scale="YlOrRd",
                     range_x=[0, 100])
        fig.update_layout(yaxis=dict(autorange="reversed"), height=300,
                          coloraxis_showscale=False, margin=dict(t=6, b=6))
        st.plotly_chart(fig, use_container_width=True)
        if len(top):
            lead = top.iloc[0]
            st.markdown(
                f"**Biggest spreader: @{lead['user']}** — score "
                f"**{lead['spreader_score']:.0f}/100**, {int(lead['hate'])} hate / "
                f"{int(lead['offensive'])} offensive posts."
            )

    st.markdown("**Full ranking**")
    show = ranking[["rank", "user", "posts", "hate", "offensive",
                    "hate_ratio", "influence", "spreader_score"]]
    st.dataframe(show, use_container_width=True, hide_index=True, height=340)
    st.caption("spreader_score = influence^w × toxicity^(1−w), rescaled to 0–100. "
               "Influence blends PageRank, degree and betweenness centrality.")
    st.download_button("Download ranking (CSV)",
                       ranking.to_csv(index=False).encode(),
                       "spreader_ranking.csv", "text/csv")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    inject_theme()          # paint the current light/dark skin first
    render_sidebar()        # holds the "Dark mode" toggle
    tabs = st.tabs(["Overview", "Data", "Train & Evaluate",
                    "Classify", "Network & Spreaders"])
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_data()
    with tabs[2]:
        render_train()
    with tabs[3]:
        render_classify()
    with tabs[4]:
        render_network()


if __name__ == "__main__":
    main()
