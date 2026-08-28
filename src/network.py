"""
Network analysis: turn classified posts into a hate-speech spreader ranking.

This is the module that makes the project novel. Instead of stopping at
"is this text hateful?", we ask "WHO in this network is driving the hate?".

Pipeline
--------
1. build_network()      : posts -> directed graph  (author --mentions--> target)
2. user_activity()      : per-user post counts + toxicity intensity
3. centrality_scores()  : PageRank / betweenness / degree per user
4. spreader_ranking()   : combine INFLUENCE x TOXICITY -> ranked spreaders

Spreader score (the key idea)
-----------------------------
A hate-speech *spreader* must be BOTH toxic AND influential:

    influence  = blended, min-max-normalised network centrality      (reach)
    toxicity   = weighted share of a user's posts that are toxic      (content)
    score      = influence^w * toxicity^(1-w)   (weighted geometric mean)

A viral-but-clean news account scores ~0 (toxicity 0); a nasty account nobody
listens to also scores low (influence ~0). Only accounts that are toxic *and*
central rise to the top -- which is exactly what "spreader" should mean.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .preprocessing import extract_mentions
from .data_utils import LABELS

# How "toxic" one post of each class is, on a 0..1 scale.
_TOXIC_INTENSITY = {"hate": 1.0, "offensive": 0.5, "normal": 0.0}


# --------------------------------------------------------------------------- #
#  1. Graph construction
# --------------------------------------------------------------------------- #
def _row_mentions(row) -> list[str]:
    """Get mention handles for a row: prefer the `mentions` column, else parse text."""
    raw = str(row.get("mentions", "") or "").strip()
    if raw:
        return [m.strip().lstrip("@").lower() for m in raw.split(",") if m.strip()]
    return extract_mentions(row.get("text", ""))


def build_network(df: pd.DataFrame, label_col: str = "label") -> nx.DiGraph:
    """Build a directed graph: edge author -> mentioned-user.

    `label_col` chooses the ground-truth "label" or the model's
    "predicted_label" column, so you can analyse either.

    Node attrs : posts, hate, offensive, normal
    Edge attrs : weight (total mentions), toxic_weight (mentions in toxic posts)
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        author = str(row["user"]).lstrip("@").lower()
        label = row.get(label_col, "normal")
        if label not in _TOXIC_INTENSITY:
            label = "normal"

        if not G.has_node(author):
            G.add_node(author, posts=0, hate=0, offensive=0, normal=0)
        G.nodes[author]["posts"] += 1
        G.nodes[author][label] += 1

        for target in _row_mentions(row):
            if target == author:
                continue
            if not G.has_node(target):
                G.add_node(target, posts=0, hate=0, offensive=0, normal=0)
            if G.has_edge(author, target):
                G[author][target]["weight"] += 1
            else:
                G.add_edge(author, target, weight=1, toxic_weight=0)
            if label in ("hate", "offensive"):
                G[author][target]["toxic_weight"] += 1

    return G


def graph_summary(G: nx.DiGraph) -> dict:
    """Headline stats for the UI's metric cards."""
    n = G.number_of_nodes()
    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4) if n > 1 else 0.0,
        "weak_components": nx.number_weakly_connected_components(G) if n else 0,
        "avg_out_degree": round(sum(d for _, d in G.out_degree()) / n, 2) if n else 0.0,
    }


# --------------------------------------------------------------------------- #
#  2. Per-user activity / toxicity
# --------------------------------------------------------------------------- #
def user_activity(G: nx.DiGraph) -> pd.DataFrame:
    """Per-user post counts and a 0..1 toxicity intensity from node attributes."""
    rows = []
    for node, a in G.nodes(data=True):
        posts = a.get("posts", 0)
        hate, off = a.get("hate", 0), a.get("offensive", 0)
        intensity = (_TOXIC_INTENSITY["hate"] * hate +
                     _TOXIC_INTENSITY["offensive"] * off)
        rows.append({
            "user": node,
            "posts": posts,
            "hate": hate,
            "offensive": off,
            "normal": a.get("normal", 0),
            "hate_ratio": round((hate + off) / posts, 3) if posts else 0.0,
            "toxicity": round(intensity / posts, 3) if posts else 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  3. Centrality
# --------------------------------------------------------------------------- #
def centrality_scores(G: nx.DiGraph) -> pd.DataFrame:
    """PageRank, betweenness and degree centrality per node.

    * PageRank is weighted by interaction counts (influence flows along mentions).
    * Betweenness is unweighted (edge counts are strengths, not distances).
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["user", "pagerank", "betweenness",
                                     "in_degree", "degree"])

    try:
        pagerank = nx.pagerank(G, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pagerank = nx.pagerank(G, weight="weight", max_iter=1000, tol=1e-4)

    betweenness = nx.betweenness_centrality(G)          # unweighted on purpose
    in_deg = nx.in_degree_centrality(G)
    deg = nx.degree_centrality(G)                        # total (in+out) for a DiGraph

    return pd.DataFrame([{
        "user": n,
        "pagerank": round(pagerank.get(n, 0.0), 5),
        "betweenness": round(betweenness.get(n, 0.0), 5),
        "in_degree": round(in_deg.get(n, 0.0), 5),
        "degree": round(deg.get(n, 0.0), 5),
    } for n in G.nodes()])


# --------------------------------------------------------------------------- #
#  4. Spreader ranking  (INFLUENCE x TOXICITY)
# --------------------------------------------------------------------------- #
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def spreader_ranking(df: pd.DataFrame,
                     label_col: str = "label",
                     influence_weight: float = 0.5) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Rank users by how much they drive hate speech through the network.

    `influence_weight` (0..1) trades off reach vs. content:
        1.0 -> pure network influence, 0.0 -> pure content toxicity, 0.5 -> balanced.

    Returns (ranking_df, graph). ranking_df is sorted by spreader_score (0..100).
    """
    G = build_network(df, label_col=label_col)
    activity = user_activity(G)
    central = centrality_scores(G)
    merged = activity.merge(central, on="user", how="outer").fillna(0.0)

    # Blend three normalised centrality measures into one "influence" number.
    influence = (0.5 * _minmax(merged["pagerank"]) +
                 0.3 * _minmax(merged["degree"]) +
                 0.2 * _minmax(merged["betweenness"]))
    merged["influence"] = _minmax(influence).round(3)   # renormalise to 0..1

    w = float(np.clip(influence_weight, 0.0, 1.0))
    # Weighted geometric mean; +epsilon avoids 0^0 and keeps it smooth.
    eps = 1e-9
    score = (np.power(merged["influence"] + eps, w) *
             np.power(merged["toxicity"] + eps, 1.0 - w))
    # Zero-out users who are either totally clean or totally isolated.
    score = np.where((merged["toxicity"] <= 0) | (merged["influence"] <= 0), 0.0, score)
    merged["spreader_score"] = (_minmax(pd.Series(score, index=merged.index)) * 100).round(1)

    cols = ["user", "posts", "hate", "offensive", "hate_ratio", "toxicity",
            "pagerank", "betweenness", "degree", "influence", "spreader_score"]
    ranking = (merged[cols]
               .sort_values(["spreader_score", "toxicity", "influence"],
                            ascending=False)
               .reset_index(drop=True))
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking, G


# --------------------------------------------------------------------------- #
#  5. Interactive visualisation (Plotly)
# --------------------------------------------------------------------------- #
def build_plotly_figure(G: nx.DiGraph,
                        ranking: pd.DataFrame,
                        top_k: int = 5,
                        seed: int = 42):
    """Return an interactive Plotly figure of the network.

    Node colour  = spreader_score (redder = bigger spreader)
    Node size    = total degree (reach)
    Top-k spreaders are outlined and always labelled.
    """
    import plotly.graph_objects as go

    if G.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=seed, k=0.9 / np.sqrt(max(G.number_of_nodes(), 1)))
    score_map = dict(zip(ranking["user"], ranking["spreader_score"]))
    top_users = set(ranking.head(top_k)["user"])

    # --- edges ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.6, color="rgba(150,150,160,0.4)"),
        hoverinfo="none", showlegend=False,
    )

    # --- nodes ---
    node_x, node_y, sizes, colors, texts, labels = [], [], [], [], [], []
    for n, a in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        deg = G.degree(n)
        sizes.append(12 + 3 * deg)
        sc = score_map.get(n, 0.0)
        colors.append(sc)
        labels.append(n if n in top_users else "")
        texts.append(
            f"<b>@{n}</b><br>spreader score: {sc:.1f}"
            f"<br>posts: {a.get('posts', 0)} "
            f"(hate {a.get('hate', 0)}, off {a.get('offensive', 0)})"
            f"<br>degree: {deg}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=labels, textposition="top center",
        textfont=dict(size=11, color="#222"),
        hovertext=texts, hoverinfo="text",
        marker=dict(
            size=sizes, color=colors, colorscale="YlOrRd", cmin=0, cmax=100,
            line=dict(width=1, color="#555"),
            colorbar=dict(title="Spreader<br>score", thickness=14),
        ),
        showlegend=False,
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white", height=560,
    )
    return fig
