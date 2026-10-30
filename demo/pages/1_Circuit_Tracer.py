"""
Circuit Tracer — hallucination detection on 10 HaluEval examples.

For each example: the model's answer, every detector from the project
(SelfCheckGPT, semantic entropy, semantic-entropy probes, naive log-prob),
and a circuit-tracer-style attribution graph showing which layers and
which prompt tokens drove the most suspicious tokens of the answer.

All numbers are pre-computed by demo/build_circuit_examples.py (GPU);
this page only reads demo/circuit_examples.json.
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Circuit Tracer", layout="wide")

DATA_FILE = Path(__file__).resolve().parents[1] / "circuit_examples.json"

# Diverging pair for attribution sign, sequential orange for surprisal.
PROMOTE = "#e34948"   # pushes the generated token up
SUPPRESS = "#2a78d6"  # pushes it down (toward the alternative)
NEUTRAL = "#9a9890"
CAT_COLORS = {
    "detected hallucination": "#e34948",
    "confident hallucination (missed)": "#eda100",
    "correct and confident": "#1baf7a",
    "hand-picked": "#6250d6",
}


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def esc(text: str) -> str:
    return html.escape(text).replace("\n", "↵<br>")


def short_token(text: str, limit: int = 12) -> str:
    t = text.replace("\n", "↵").replace("<|", "⟨").replace("|>", "⟩")
    t = t if t.strip() else repr(t)
    return t if len(t) <= limit else t[: limit - 1] + "…"


def surprisal_color(nats: float) -> str:
    t = min(max(nats / 6.0, 0.0), 1.0)
    lo, hi = np.array([253, 240, 230]), np.array([235, 104, 52])
    r, g, b = (lo + (hi - lo) * t).astype(int)
    return f"rgb({r},{g},{b})"


def badge(text: str, color: str) -> str:
    return (f"<span style='background:{color};color:white;padding:2px 10px;"
            f"border-radius:10px;font-size:0.85em;font-weight:600'>{esc(text)}</span>")


# ---------------------------------------------------------------------------
# Attribution graph
# ---------------------------------------------------------------------------

def columns_for(ex: dict, tgt: dict) -> list:
    """Token positions to draw (with '…' gaps), keeping the graph readable."""
    P, q = ex["prompt_len"], tgt["position_q"]
    tokens = ex["tokens"]
    user = [i for i in range(P) if tokens[i]["role"] == "user"]
    node_pos = {n["pos"] for n in tgt["nodes"]}

    keep = set(node_pos) | {q} | set(range(max(P, q - 6), q + 1))
    if len(user) <= 35:
        keep |= set(user)
    else:
        for p in node_pos:
            keep |= {p - 1, p, p + 1}
    keep = sorted(p for p in keep if 0 <= p <= q)

    cols, prev = [], None
    for p in keep:
        if prev is not None and p - prev > 1:
            cols.append(None)  # gap
        cols.append(p)
        prev = p
    return cols


def node_hover(n: dict, tokens: list, q: int) -> str:
    v = n["value"]
    up = ", ".join(esc(short_token(t, 14)) for t, _ in n["promotes"]["up"])
    down = ", ".join(esc(short_token(t, 14)) for t, _ in n["promotes"]["down"])
    if n["type"] == "attn":
        head_txt = "<br>".join(
            f"  head {h}: {c:+.2f} (attends {a:.0%})" for h, c, a in n.get("heads", []))
        title = f"<b>L{n['layer']} attention ← '{esc(short_token(tokens[n['pos']]['text'], 20))}'</b>"
        body = f"moves info from that token to the predicting position<br>{head_txt}"
    elif n["type"] == "mlp":
        title = f"<b>L{n['layer']} MLP @ '{esc(short_token(tokens[q]['text'], 20))}'</b>"
        body = "computed at the predicting position"
    else:
        title = f"<b>Token embedding '{esc(short_token(tokens[q]['text'], 20))}'</b>"
        body = "direct path from the input token"
    return (f"{title}<br>Δlogit contribution: <b>{v:+.3f}</b><br>{body}"
            f"<br><br>output card — promotes: {up}<br>suppresses: {down}")


def graph_figure(ex: dict, tgt: dict, n_layers: int) -> go.Figure:
    tokens, q = ex["tokens"], tgt["position_q"]
    cols = columns_for(ex, tgt)
    x_of = {p: k for k, p in enumerate(cols) if p is not None}
    labels = ["…" if p is None else short_token(tokens[p]["text"]) for p in cols]

    nodes = tgt["nodes"]
    vmax = max((abs(n["value"]) for n in nodes), default=1.0) or 1.0
    y_tok, y_emb, y_out = -1.3, -0.6, n_layers + 0.7
    xq = x_of[q]

    def xy(n):
        if n["type"] == "attn":
            return x_of[n["pos"]], n["layer"]
        if n["type"] == "mlp":
            return xq + 0.32, n["layer"]
        return xq, y_emb

    fig = go.Figure()

    # highlight the predicting column
    fig.add_vrect(x0=xq - 0.5, x1=xq + 0.8, fillcolor="rgba(235,104,52,0.10)", line_width=0, layer="below")

    # edges: token -> attention node (straight), node -> output (curved)
    for n in sorted(nodes, key=lambda n: abs(n["value"])):
        x, y = xy(n)
        w = abs(n["value"]) / vmax
        color = PROMOTE if n["value"] > 0 else SUPPRESS
        alpha = 0.18 + 0.6 * w
        rgba = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha:.2f})"
        if n["type"] == "attn":
            fig.add_trace(go.Scatter(x=[x, x], y=[y_tok, y], mode="lines", hoverinfo="skip",
                                     line=dict(color=rgba, width=0.6 + 3 * w), showlegend=False))
        t = np.linspace(0, 1, 24)
        cx, cy = (x + xq) / 2, max(y, y_out) - 0.2
        bx = (1 - t) ** 2 * x + 2 * (1 - t) * t * cx + t ** 2 * xq
        by = (1 - t) ** 2 * y + 2 * (1 - t) * t * cy + t ** 2 * y_out
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", hoverinfo="skip",
                                 line=dict(color=rgba, width=0.6 + 3.5 * w), showlegend=False))

    # nodes, one trace per kind so the legend explains the shapes
    for kind, symbol, name in [("attn", "circle", "attention (head × source token)"),
                               ("mlp", "square", "MLP"), ("embed", "diamond", "token embedding")]:
        group = [n for n in nodes if n["type"] == kind]
        if not group:
            continue
        xs, ys = zip(*[xy(n) for n in group])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=name,
            marker=dict(symbol=symbol, size=[9 + 22 * abs(n["value"]) / vmax for n in group],
                        color=[PROMOTE if n["value"] > 0 else SUPPRESS for n in group],
                        line=dict(color="white", width=1.5)),
            hovertext=[node_hover(n, tokens, q) for n in group], hoverinfo="text",
        ))

    # output node
    fig.add_trace(go.Scatter(
        x=[xq], y=[y_out], mode="markers+text", name="output logit",
        marker=dict(symbol="star", size=24, color="#0b0b0b"),
        text=[f"  '{short_token(tgt['token'], 16)}' over '{short_token(tgt['alt'], 16)}'"],
        textposition="middle right", textfont=dict(size=13),
        hovertext=[f"<b>Δlogit = {tgt['logit_diff']:+.2f}</b><br>p('{esc(tgt['token'])}') = {tgt['p_token']:.3f}"
                   f"<br>p('{esc(tgt['alt'])}') = {tgt['p_alt']:.3f}"],
        hoverinfo="text",
    ))
    # legend entries for the sign colors
    for color, name in [(PROMOTE, "pushes generated token ↑"), (SUPPRESS, "pushes alternative ↑")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=10, color=color)))

    layer_ticks = list(range(0, n_layers, 2))
    fig.update_layout(
        height=140 + 27 * n_layers, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickvals=list(range(len(cols))), ticktext=labels, tickangle=-60,
                   showgrid=False, zeroline=False, range=[-0.8, len(cols) + 3]),
        yaxis=dict(tickvals=[y_emb] + layer_ticks + [y_out],
                   ticktext=["embed"] + [f"L{l}" for l in layer_ticks] + ["output"],
                   range=[y_tok - 0.2, y_out + 0.8], zeroline=False, gridcolor="rgba(128,128,128,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(align="left"),
    )
    return fig


def lens_figure(tgt: dict) -> go.Figure:
    lens = pd.DataFrame(tgt["lens"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lens["layer"], y=lens["p_token"], mode="lines+markers",
                             name=f"p('{short_token(tgt['token'], 16)}')  generated", line=dict(color=PROMOTE, width=2),
                             hovertemplate="layer %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=lens["layer"], y=lens["p_alt"], mode="lines+markers",
                             name=f"p('{short_token(tgt['alt'], 16)}')  best alternative", line=dict(color=SUPPRESS, width=2),
                             hovertemplate="layer %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="layer (residual stream read through the final norm + unembedding)",
                      yaxis_title="probability", yaxis_range=[0, 1.02],
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

data = load_data()
st.title("Hallucination circuit tracer")
st.caption(
    "10 HaluEval questions answered by Qwen2.5-0.5B-Instruct. Pick one to see what every "
    "detector in the project says about the answer, then trace *which layers and prompt tokens* "
    "produced its most suspicious tokens."
)

if data is None:
    st.error(
        "`demo/circuit_examples.json` not found. Build it on the server (GPU, a few minutes):\n\n"
        "```\nsbatch slurm/demo_circuits.sbatch\n```\n"
        "then copy `demo/circuit_examples.json` back next to this app."
    )
    st.stop()

meta, examples = data["meta"], data["examples"]
n_layers = meta["n_layers"]


def option_label(i: int) -> str:
    ex = examples[i]
    prompt = ex["prompt"].replace("\n", " ")
    return f"{i + 1}. {ex['subset'].upper()} · {ex['category']} — {prompt[:70]}{'…' if len(prompt) > 70 else ''}"


choice = st.selectbox("HaluEval example", range(len(examples)), format_func=option_label)
ex = examples[choice]
P = ex["prompt_len"]

# --- Row 1: question / answer / detectors ----------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    label = ex["label"]["hallucinated"]
    label_txt = {1: "hallucinated", 0: "faithful"}.get(label, "unlabeled")
    st.markdown(
        badge(ex["category"], CAT_COLORS.get(ex["category"], NEUTRAL)) + "&nbsp;&nbsp;"
        + badge(f"provisional label: {label_txt}", PROMOTE if label == 1 else "#1baf7a"),
        unsafe_allow_html=True,
    )
    st.markdown("**Prompt**")
    st.info(ex["prompt"])
    refs = "; ".join(ex["reference_correct"])
    st.markdown(f"**Reference answer:** {esc(refs)}", unsafe_allow_html=True)
    if ex.get("context"):
        with st.expander("Knowledge the dataset provides (the model was NOT shown this)"):
            st.write(ex["context"])

    st.markdown("**Model answer**")
    target_idx = {t["answer_index"] for t in ex["targets"]}
    tab_tok, tab_sent = st.tabs(["Tokens · surprisal", "Sentences · SelfCheck-NLI"])
    with tab_tok:
        spans = []
        for i, t in enumerate(ex["answer_tokens"]):
            s = -t["logprob"]
            alts = " | ".join(f"{a!r} {p:.2f}" for a, p in t["top"])
            border = "border:2px solid #0b0b0b;" if i in target_idx else ""
            spans.append(
                f"<span title='p={np.exp(t['logprob']):.3f}  surprisal={s:.2f}  entropy={t['entropy']:.2f}&#10;top: {html.escape(alts)}' "
                f"style='background:{surprisal_color(s)};color:#111;padding:1px 2px;border-radius:3px;{border}'>"
                f"{esc(t['text'])}</span>")
        st.markdown(f"<div style='line-height:2.0;font-size:1.02em'>{''.join(spans)}</div>", unsafe_allow_html=True)
        st.caption("Darker orange = more surprising to the model. Boxed tokens have an attribution graph below. Hover a token for its probability and top alternatives.")
    with tab_sent:
        for s in ex["sentences"]:
            score = s["selfcheck_nli"]
            color = PROMOTE if score >= 0.5 else ("#eda100" if score >= 0.25 else "#1baf7a")
            st.markdown(
                f"<div style='border-left:6px solid {color};padding:4px 10px;margin:4px 0'>"
                f"<b>{score:.2f}</b> &nbsp; {esc(s['text'])}</div>", unsafe_allow_html=True)
        st.caption("SelfCheck-NLI: average P(contradiction) between the sentence and each of the 15 re-sampled answers. "
                   "High = the model does not reproduce this claim when asked again.")

with right:
    st.markdown("**Detectors**")
    d = ex["detectors"]

    def row(name, value_txt, pct, help_txt):
        st.markdown(f"**{name}** &nbsp; `{value_txt}`", unsafe_allow_html=True)
        if pct is not None:
            st.progress(float(pct), text=f"higher risk than {pct:.0%} of all 24,652 answers")
        st.caption(help_txt)

    row("SelfCheckGPT-NLI (B3)", f"{d['selfcheck_nli']['value']:.2f}", None,
        "Mean sentence contradiction vs 15 samples (0 = consistent, 1 = contradicted).")
    nc = d["n_clusters"]
    row("Semantic entropy (B4)", f"{d['semantic_entropy']['value']:.2f} · {nc['value']}/{nc['of']} meaning clusters",
        d["semantic_entropy"]["pct"], "Entropy over meaning-clusters of 15 samples. More clusters = less agreement.")
    row("Semantic-entropy probe (B5)", f"{d['sep_tbg']['value']:.2f}", d["sep_tbg"]["pct"],
        "Linear probe on the hidden state BEFORE generating; predicts high semantic entropy in one pass.")
    row("Naive log-prob (B6)", f"{d['logprob_neg_mean']['value']:.2f}", d["logprob_neg_mean"]["pct"],
        "−mean token log-probability of the answer"
        + (f"; B6 Youden threshold = {d['logprob_youden_threshold']:.2f}" if d.get("logprob_youden_threshold") else "") + ".")
    row("Accuracy probe (B5 baseline)", f"{d['accprobe_tbg']['value']:.2f}", d["accprobe_tbg"]["pct"],
        "Probe trained directly on the provisional labels (can learn the question type, not just hallucination).")

st.divider()

# --- Row 2: attribution graph -------------------------------------------------
st.subheader("Attribution graph")
targets = ex["targets"]
if not targets:
    st.info("This answer has no content tokens to trace.")
    st.stop()
pick = st.radio(
    "Explain this answer token",
    range(len(targets)), horizontal=True,
    format_func=lambda k: f"'{short_token(targets[k]['token'], 18)}'  (p={targets[k]['p_token']:.2f})",
)
tgt = targets[pick]

g_col, card_col = st.columns([3, 1], gap="medium")
with g_col:
    st.plotly_chart(graph_figure(ex, tgt, n_layers), width="stretch")
    st.caption(
        f"Why did the model write **'{tgt['token']}'** instead of **'{tgt['alt']}'**? "
        f"Δlogit = {tgt['logit_diff']:+.2f} is split exactly across the token embedding, every MLP, and every "
        f"attention head × source token; the strongest nodes covering {tgt['covered']:.0%} of the total are drawn. "
        "Red pushes toward the generated token, blue toward the alternative. Vertical lines show which prompt "
        "token an attention node read from. Hover a node for its heads and the tokens it promotes."
    )
with card_col:
    st.markdown("**Next-token distribution**")
    top = pd.DataFrame(tgt["top_next"], columns=["token", "p"])
    top["token"] = top["token"].map(lambda t: repr(t))
    st.dataframe(top.style.format({"p": "{:.3f}"}).bar(subset=["p"], color="#eb6834", vmin=0, vmax=1),
                 hide_index=True, width="stretch")
    st.markdown("**Strongest nodes**")
    tokens = ex["tokens"]
    rows = []
    for n in sorted(tgt["nodes"], key=lambda n: -abs(n["value"]))[:8]:
        where = ("emb" if n["type"] == "embed" else
                 f"L{n['layer']} {'attn ← ' + repr(short_token(tokens[n['pos']]['text'], 14)) if n['type'] == 'attn' else 'MLP'}")
        rows.append({"node": where, "Δlogit": n["value"],
                     "promotes": ", ".join(short_token(t, 10) for t, _ in n["promotes"]["up"][:3])})
    st.dataframe(pd.DataFrame(rows).style.format({"Δlogit": "{:+.2f}"}), hide_index=True, width="stretch")

st.markdown("**Logit lens — at which layer does the model commit?**")
st.plotly_chart(lens_figure(tgt), width="stretch")
st.caption("Each layer's residual stream at the predicting position, decoded with the final norm and unembedding. "
           "A fact the model 'knows' tends to appear in middle layers; a guessed token often only wins in the last few.")

st.divider()

# --- Row 3: semantic clusters ------------------------------------------------
st.subheader("What the model says when asked again (15 samples → meaning clusters)")
sizes = [len(c) for c in ex["clusters"]]
fig = go.Figure(go.Bar(x=[f"C{i + 1}" for i in range(len(sizes))], y=sizes, marker_color="#2a78d6",
                       hovertemplate="%{x}: %{y} samples<extra></extra>"))
fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="samples", xaxis_title="cluster")
st.plotly_chart(fig, width="stretch")
for i, cluster in enumerate(ex["clusters"][:5]):
    with st.expander(f"Cluster C{i + 1} · {len(cluster)} sample(s)"):
        for s in cluster[:3]:
            st.markdown(f"> {s}")

with st.expander("How this works and what it is not"):
    st.markdown(f"""
- **Model:** `{meta['model']}` ({n_layers} layers, {meta['n_heads']} heads). Every answer here is the model's real A2 output.
- **Attribution graph:** the logit difference between the generated token and its strongest alternative is decomposed
  *exactly* (final RMSNorm scale frozen) into the embedding, each layer's MLP output and each attention head's output split
  by the source token it attended to. This is a **direct-effect** graph in the style of Anthropic's circuit-tracer
  attribution graphs. It is not their method: circuit-tracer replaces MLPs with transcoders to get interpretable
  *features*, and no transcoders exist for this model, so nodes here are model components (layers, heads), and
  indirect paths between nodes are not traced.
- **Suspicious tokens:** the most surprising content tokens of the sentence SelfCheck-NLI distrusts most.
- **Labels are provisional** (NLI against the dataset references); treat them as a hint, not ground truth.
- Built `{meta['created']}` at commit `{meta['git_commit_sha'][:7]}`.
""")
