#!/usr/bin/env python3
"""Render exact Transformer SVG figures using only the Python standard library.

  python3 -B scripts/render_transformer_figures.py --check
  python3 -B scripts/render_transformer_figures.py --output content/posts/ai/transformer-attention/assets

The check uses direct high-precision Decimal softmax as an independent oracle
for the renderer's shifted float calculation, and checks symbolic matrix shapes,
head feature intervals and actual SVG arrow records. No tensor package or SVG
renderer is required. The source convention is PyTorch Linear's [out,in] weight,
so projection formulas use W transpose plus bias.

Attention reference: https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2
"""
import argparse
from decimal import Decimal, localcontext
from html import escape
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "content/posts/ai/transformer-attention/assets"
NS = {"s": "http://www.w3.org/2000/svg"}
COLORS = {
    "blue": ("#edf5ff", "#7299c6"),
    "purple": ("#f3effb", "#9b87bb"),
    "orange": ("#fff3e5", "#c49a6b"),
    "green": ("#edf7ef", "#83aa90"),
    "gray": ("#f7f9fc", "#c9d2df"),
}
CAUSAL_SCORES = (
    ("1.2", ".7", "-.3", ".8", "-1.1"),
    ("-.4", "1.5", ".6", "-.2", ".3"),
    (".1", "-.6", "1.1", ".4", "-.7"),
    (".3", ".2", "-.5", "1.3", ".9"),
    ("-.2", ".4", ".1", "-.3", "1.0"),
)
MOE_LOGITS = ((2, 1, 0, -1), (0, 3, 1, 2),
              (1, -1, 3, 0), (-1, 0, 1, 3))


def shape(dims):
    return "[" + ",".join(map(str, dims)) + "]"


def number(value):
    return "−∞" if value is None else f"{value:.3f}"


def softmax_records():
    rows = []
    for name, scores in (("Unmasked", (2.0, 1.0, 0.0)),
                         ("Third key masked", (2.0, 1.0, None))):
        maximum = max(x for x in scores if x is not None)
        shifted = tuple(None if x is None else x - maximum for x in scores)
        exponents = tuple(0.0 if x is None else math.exp(x) for x in shifted)
        total = math.fsum(exponents)
        probabilities = tuple(x / total for x in exponents)
        rows.append({"name": name, "scores": scores, "maximum": maximum,
                     "shifted": shifted, "exponents": exponents,
                     "total": total, "probabilities": probabilities})
    return rows


def mha_records():
    records = {"C": 48, "H": 8, "Dh": 6,
               "input": ("B", "T", 48), "weight": (48, 48), "bias": (48,),
               "projected": ("B", "T", 48),
               "reshaped": ("B", "T", 8, 6), "permutation": (0, 2, 1, 3),
               "split": ("B", 8, "T", 6), "selected": ("B", "T", 6),
               "scores": ("B", "T", "T"), "head_output": ("B", "T", 6),
               "merged": ("B", "T", 48), "output": ("B", "T", 48),
               "concat_axis": -1}
    records["concat"] = [{"head": i + 1, "shape": ("B", "T", 6),
                           "start": i * 6, "stop": (i + 1) * 6} for i in range(8)]
    return records


def causal_records():
    with localcontext() as context:
        context.prec = 70
        scores = tuple(tuple(Decimal(value) for value in row) for row in CAUSAL_SCORES)
        masked, weights = [], []
        for i, row in enumerate(scores):
            masked_row = tuple(value if j <= i else None for j, value in enumerate(row))
            maximum = max(value for value in masked_row if value is not None)
            terms = tuple(Decimal(0) if value is None else (value - maximum).exp()
                          for value in masked_row)
            masked.append(masked_row)
            weights.append(tuple(value / sum(terms) for value in terms))
    return {"scores": scores, "masked": masked, "weights": tuple(weights),
            "rule": "Key j is allowed iff j <= Query i", "precision_digits": 70}


def moe_records():
    tokens = []
    with localcontext() as context:
        context.prec = 70
        for t, values in enumerate(MOE_LOGITS, 1):
            logits = tuple(Decimal(value) for value in values)
            selected = sorted(range(4), key=lambda i: (-logits[i], i))[:2]
            terms = [(logits[i] - logits[selected[0]]).exp() for i in selected]
            weights = [value / sum(terms) for value in terms]
            chosen = [{"expert": i + 1, "weight": weight, "gain": i + 1}
                      for i, weight in zip(selected, weights)]
            gain = sum(entry["weight"] * entry["gain"] for entry in chosen)
            x = tuple(Decimal(int(i == t - 1)) for i in range(4))
            tokens.append({"token": t, "x": x, "logits": logits,
                           "chosen": chosen, "gain": gain,
                           "output": tuple(gain * value for value in x)})
    return {"tokens": tokens, "top_k": 2, "expert_gains": (1, 2, 3, 4),
            "router_weight": tuple(tuple(Decimal(row[j]) for row in MOE_LOGITS) for j in range(4)),
            "construction": "x_t=e_t in R4; router W_r transpose equals the displayed logits rows; E_j(x)=j*x"}


def scoring_records():
    return {"methods": ("additive", "dot", "scaled", "cosine"),
            "Q": ("B", "Tq", "Dh"), "K": ("B", "Tk", "Dh"),
            "V": ("B", "Tk", "Dv"), "scores": ("B", "Tq", "Tk"),
            "weights": ("B", "Tq", "Tk"), "output": ("B", "Tq", "Dv"),
            "column_vectors": {"q_i": ("Dh", 1), "k_j": ("Dh", 1),
                               "W_q": ("A", "Dh"), "W_k": ("A", "Dh"),
                               "v_a": ("A", 1), "b": ("A", 1)},
            "cosine_requires_nonzero_vectors": True}


class SVG:
    def __init__(self, width, height, title, description, records):
        self.width, self.height = width, height
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title>',
                      f'<desc id="desc">{escape(description)}</desc>',
                      '<metadata id="calculation-records">' + escape(json.dumps(records, ensure_ascii=False, default=str), quote=False) + '</metadata>',
                      '''<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#536174"/></marker>
  <style>
    text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #233148; }
    .title { font-size: 32px; font-weight: 700; }
    .subtitle { font-size: 18px; fill: #475569; }
    .heading { font-size: 20px; font-weight: 700; }
    .label { font-size: 18px; font-weight: 700; }
    .value { font-family: "DejaVu Sans Mono", monospace; font-size: 21px; }
    .small { font-size: 16px; fill: #475569; }
    .tiny { font-size: 14px; fill: #475569; }
    .formula { font-size: 23px; }
    .arrow { stroke: #536174; stroke-width: 2; fill: none; stroke-linejoin: round; marker-end: url(#arrow); }
  </style>
</defs>''',
                      f'<rect width="{width}" height="{height}" fill="#ffffff"/>']

    def text(self, x, y, value, cls="label", anchor="middle", ident=None, size=None):
        attrs = f' id="{ident}"' if ident else ""
        attrs += f' style="font-size:{size}px"' if size else ""
        self.items.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}"{attrs}>{escape(str(value))}</text>')

    def box(self, x, y, width, height, color="gray", ident=None, radius=12, fill=None):
        background, stroke = COLORS[color]
        attrs = f' id="{ident}"' if ident else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill or background}" stroke="{stroke}" stroke-width="1.6"{attrs}/>')

    def arrow(self, points, source=None, target=None, dashed=False):
        command = "M" + " L".join(f"{x},{y}" for x, y in points)
        attrs = f' data-from="{source}" data-to="{target}"' if source else ""
        attrs += ' stroke-dasharray="5 4"' if dashed else ""
        self.items.append(f'<path d="{command}" class="arrow"{attrs}/>')

    def finish(self):
        return "\n".join(self.items + ["</svg>", ""])


def softmax_svg(rows):
    drawing = SVG(1400, 770, "Stable row-wise softmax: unmasked and masked examples",
                  "For scores 2, 1, 0, subtract 2, exponentiate, sum and normalize to approximately 0.665, 0.245, 0.090. Masking the third key gives 0.731, 0.269, 0.000. Each row sums to one; displayed values are rounded.", rows)
    drawing.text(48, 49, "Stable row-wise Softmax", "title", "start")
    drawing.text(48, 82, "One query row · normalize over the key positions (last axis)", "subtitle", "start")
    columns = [(48, 160, "Scores z", "purple"),
               (242, 210, "1  Subtract max", "blue"),
               (486, 210, "2  Exponentiate", "orange"),
               (730, 194, "3  Sum", "orange"),
               (960, 392, "4  Normalize", "green")]
    for row_index, row in enumerate(rows):
        top = 112 + row_index * 286
        drawing.box(32, top, 1336, 266, "gray", radius=16, fill="#ffffff")
        prefix = "A" if row_index == 0 else "B"
        vector = ", ".join(number(x) for x in row["scores"])
        drawing.text(51, top + 31, f"{prefix} · {row['name']}: z = [{vector}]", "heading", "start")
        card_y = top + 52
        for x, width, title, color in columns:
            drawing.box(x, card_y, width, 194, color)
            drawing.text(x + width / 2, card_y + 29, title, "label")
        for (x, width, _, _), (next_x, _, _, _) in zip(columns, columns[1:]):
            drawing.arrow([(x + width + 3, card_y + 103), (next_x - 4, card_y + 103)])
        for index in range(3):
            baseline = card_y + 75 + index * 38
            drawing.text(128, baseline, number(row["scores"][index]), "value")
            drawing.text(347, baseline, number(row["shifted"][index]), "value")
            drawing.text(591, baseline, number(row["exponents"][index]), "value")
        drawing.text(347, card_y + 49, "m = " + number(row["maximum"]), "tiny")
        drawing.text(591, card_y + 49, "exp(z − m)", "tiny")
        exponent_labels = [number(x) for x in row["exponents"]]
        drawing.text(827, card_y + 83, exponent_labels[0] + " + " + exponent_labels[1], "small")
        drawing.text(827, card_y + 117, "+ " + exponent_labels[2], "small")
        drawing.text(827, card_y + 162, "Z = " + number(row["total"]), "label")
        probability_labels = [number(x) for x in row["probabilities"]]
        for index, probability in enumerate(probability_labels):
            equation = f"{exponent_labels[index]} / {number(row['total'])} → {probability}"
            drawing.text(1156, card_y + 67 + index * 34, equation, "value", size=18)
        drawing.text(1156, card_y + 175, " + ".join(probability_labels) + " = 1.000", "label", ident=f"row-{row_index}-sum", size=16)
    drawing.text(48, 713, "Displayed values are rounded to 3 decimals; normalization uses unrounded exponentials.", "small", "start")
    drawing.text(48, 744, "Masked entry: exp(−∞) = 0 exactly. Each unrounded probability row sums to 1.", "small", "start")
    return drawing.finish()


def mha_svg(records):
    drawing = SVG(1400, 1390, "Multi-head self-attention shapes: C=48, H=8, Dh=6",
                  "The same input X feeds three independent linear projections. Each projected tensor is reshaped and transposed to B,8,T,6. Every head i receives its own Qi, Ki and Vi, producing B,T,6. Eight head outputs concatenate along the last feature axis to B,T,48 before the output linear projection.", records)
    drawing.text(48, 48, "Multi-head self-attention", "title", "start")
    drawing.text(48, 81, "C = 48 · H = 8 · Dh = C / H = 6 · B = batch · T = tokens", "subtitle", "start")
    drawing.text(1352, 81, "Three independent learned projections", "small", "end")
    drawing.box(575, 110, 250, 70, "blue", "x")
    drawing.text(700, 138, "Input X", "label")
    drawing.text(700, 166, shape(records["input"]), "value")
    centers = [240, 700, 1160]
    for letter, center, color in zip("qkv", centers, ("blue", "purple", "orange")):
        upper = letter.upper()
        drawing.arrow([(700, 180), (700, 206), (center, 206), (center, 234)], "x", letter + "-project")
        drawing.box(center - 165, 234, 330, 96, color, letter + "-project")
        drawing.text(center, 263, f"{upper} = X W{letter}ᵀ + b{letter}", "heading")
        drawing.text(center, 289, f"W{letter} [48,48]; b{letter} [48]", "tiny")
        drawing.text(center, 316, shape(records["projected"]), "value")
        drawing.arrow([(center, 330), (center, 368)], letter + "-project", letter + "-split")
        drawing.box(center - 165, 368, 330, 122, color, letter + "-split")
        drawing.text(center, 398, "reshape → " + shape(records["reshaped"]), "small")
        drawing.text(center, 426, "transpose(1,2)", "small")
        drawing.text(center, 461, upper + " " + shape(records["split"]), "value")
    drawing.box(56, 550, 1288, 356, "gray", "head-template", fill="#ffffff")
    drawing.text(1308, 575, "Any head i", "label", "end")
    drawing.text(1308, 597, "i = 1…8", "small", "end")
    for letter, center, color in zip("qkv", centers, ("blue", "purple", "orange")):
        drawing.arrow([(center, 490), (center, 610)], letter + "-split", letter + "-i")
        drawing.text(center + 13, 524, "select i", "tiny", "start")
        drawing.box(center - 120, 610, 240, 62, color, letter + "-i")
        drawing.text(center, 636, letter.upper() + "_i", "heading")
        drawing.text(center, 660, shape(records["selected"]), "value", size=19)
    drawing.arrow([(240, 672), (240, 752), (370, 752)], "q-i", "attention")
    drawing.arrow([(700, 672), (700, 708)], "k-i", "attention")
    drawing.arrow([(1160, 672), (1160, 752), (1030, 752)], "v-i", "attention")
    drawing.box(370, 708, 660, 98, "purple", "attention")
    drawing.text(700, 738, "Scaled dot-product attention", "heading")
    drawing.text(700, 777, "softmax(Q_i K_iᵀ / √6 + M) V_i", "formula")
    drawing.arrow([(700, 806), (700, 834)], "attention", "head-i-output")
    drawing.box(540, 834, 320, 54, "green", "head-i-output")
    drawing.text(700, 868, "head_i " + shape(records["head_output"]), "value")
    drawing.text(80, 937, "Repeat with each head’s own Q_i, K_i, V_i", "small", "start")
    drawing.text(1320, 937, "Concat on the last feature axis (dim = −1)", "small", "end")
    drawing.arrow([(700, 888), (700, 966)], "head-family", "concat", dashed=True)
    drawing.box(72, 966, 1256, 78, "green", "concat")
    for column, head in enumerate(records["concat"]):
        x = 72 + column * 157
        # Equal-width neighboring rectangles represent six adjacent features,
        # not a stack along the token dimension.
        drawing.box(x, 966, 157, 78, "green", f"head-output-{head['head']}", radius=0,
                    fill="#edf7ef" if column % 2 == 0 else "#f7fbf8")
        drawing.text(x + 78.5, 996, "head " + str(head["head"]), "label", size=17)
        drawing.text(x + 78.5, 1027, shape(head["shape"]), "value", size=16)
    drawing.arrow([(95, 1072), (1305, 1072)])
    drawing.text(700, 1103, "6 + 6 + 6 + 6 + 6 + 6 + 6 + 6 = 48 features", "small")
    drawing.text(700, 1136, "Concat(head_1,…,head_8) → " + shape(records["merged"]), "value", size=20)
    drawing.arrow([(700, 1146), (700, 1178)], "concat", "output-linear")
    drawing.box(425, 1178, 550, 84, "green", "output-linear")
    drawing.text(700, 1209, "Output linear: Y = Concat W_oᵀ + b_o", "heading")
    drawing.text(700, 1242, shape(records["merged"]) + " → " + shape(records["output"]), "value")
    drawing.text(700, 1300, "Batched merge: [B,8,T,6] → transpose(1,2) → [B,T,8,6] → reshape → [B,T,48]", "small")
    drawing.text(700, 1333, "PyTorch W uses [out,in]. M is an optional [T,T] additive mask, broadcast over batch and heads.", "small")
    drawing.text(700, 1363, "Valid query rows shown; padding/all-masked handling omitted.", "small")
    return drawing.finish()


def causal_svg(records):
    drawing = SVG(1600, 750, "Causal mask: fixed scores, masked logits and exact row softmax",
                  "Rows are queries and columns are keys. Key j is allowed when j is at most the query index i. Future logits become negative infinity before softmax and their weights are exactly zero. The second row has weights approximately 0.130,0.870,0,0,0.", records)
    drawing.text(48, 49, "Causal masking before row-wise Softmax", "title", "start")
    drawing.text(48, 84, "Rows: Query i · Columns: Key j · Allow j ≤ i, including the current token", "subtitle", "start")
    panels = [(32, "Scores S", "Fixed illustrative scores", "purple", "scores"),
              (566, "Masked scores S + M", "Allowed: M = 0 · Blocked: M = −∞", "green", "masked"),
              (1100, "Weights A", "A = softmax(S + M, dim = −1)", "blue", "weights")]
    for left, title, subtitle, color, field in panels:
        drawing.box(left, 144, 468, 466, "gray", fill="#ffffff")
        drawing.text(left + 234, 180, title, "heading")
        drawing.text(left + 234, 207, subtitle, "small", size=15)
        grid_x, grid_y, cell_w, cell_h = left + 80, 260, 70, 56
        drawing.text(grid_x + 175, 232, "Key j →", "small")
        drawing.text(left + 28, 249, "i ↓", "small")
        for j in range(5):
            drawing.text(grid_x + (j + .5) * cell_w, 250, str(j + 1), "small")
        for i, row in enumerate(records[field]):
            drawing.text(grid_x - 21, grid_y + i * cell_h + 35, str(i + 1), "small")
            for j, value in enumerate(row):
                blocked = field != "scores" and j > i
                drawing.box(grid_x + j * cell_w, grid_y + i * cell_h,
                            cell_w, cell_h, "gray" if blocked else color, radius=0)
                drawing.text(grid_x + (j + .5) * cell_w, grid_y + i * cell_h + 35,
                             number(value), "value", ident=f"causal-{field}-{i+1}-{j+1}", size=17)
        footer = {"scores": "All 25 input scores are shown unchanged.",
                  "masked": "The diagonal and all earlier keys are allowed.",
                  "weights": "Future weights = 0 exactly; each row sums to 1."}[field]
        drawing.text(left + 234, 581, footer, "small", size=14)
    drawing.arrow([(504, 391), (562, 391)], "scores", "masked")
    drawing.text(533, 367, "add M", "tiny")
    drawing.arrow([(1038, 391), (1096, 391)], "masked", "weights")
    drawing.text(1067, 367, "softmax", "tiny")
    drawing.text(48, 652, "Fixed illustrative scores; Decimal probabilities are displayed to 3 decimals. Row sums are 1 before rounding.", "small", "start")
    second_masked = "[" + ", ".join(number(value) for value in records["masked"][1]) + "]"
    second_weights = "[" + ", ".join(number(value) for value in records["weights"][1]) + "]"
    drawing.text(48, 687, f"Row 2: softmax({second_masked}) = {second_weights}", "small", "start")
    drawing.text(48, 722, "Row i reads only keys 1…i. Blocking happens before softmax; future scores never enter its denominator.", "small", "start")
    return drawing.finish()


def moe_svg(records):
    drawing = SVG(1600, 1140, "Constructed top-two MoE routing with separate token outputs",
                  "Four one-hot token inputs route to four shared toy experts. Each token selects exactly two experts and applies softmax only to those two logits. Each lane combines its own two expert results; repeated expert labels refer to the same function E_j(x)=j x.", records)
    drawing.text(48, 48, "Top-2 MoE routing: one independent output per token", "title", "start")
    drawing.text(48, 82, "Constructed 4-token / 4-expert example · normalize only the two selected logits", "subtitle", "start")
    drawing.box(32, 107, 1536, 68, "gray", fill="#ffffff")
    drawing.text(800, 134, "x_t = e_t ∈ R⁴ · E1(x) = x · E2(x) = 2x · E3(x) = 3x · E4(x) = 4x", "small")
    drawing.text(800, 160, "The four logits rows define W_rᵀ, so r_t = x_t W_rᵀ. Repeated expert names share the same parameters.", "small")
    for center, title in ((121, "Token"), (420, "Router logits: [E1,E2,E3,E4]"),
                          (880, "Two chosen experts"), (1309, "This token’s weighted output")):
        drawing.text(center, 209, title, "label", size=17)
    for token in records["tokens"]:
        t = token["token"]
        top = 228 + (t - 1) * 210
        drawing.box(32, top, 1536, 190, "gray", fill="#ffffff")
        drawing.box(56, top + 58, 130, 72, "blue", f"token-{t}")
        drawing.text(121, top + 87, f"Token {t}", "label")
        drawing.text(121, top + 114, f"x_{t} = e_{t}", "small")
        drawing.arrow([(186, top + 94), (220, top + 94)], f"token-{t}", f"router-{t}")
        drawing.box(220, top + 40, 400, 106, "purple", f"router-{t}")
        drawing.text(420, top + 70, f"r_{t} = x_{t} W_rᵀ", "label")
        logits = "[" + ", ".join(number(value) for value in token["logits"]) + "]"
        drawing.text(420, top + 107, logits, "value", ident=f"moe-logits-{t}", size=16)
        drawing.text(420, top + 133, "Top-2 → selected-only softmax", "tiny")
        drawing.box(1110, top + 40, 398, 106, "green", f"output-{t}")
        for rank, entry in enumerate(token["chosen"]):
            expert = entry["expert"]
            center_y = top + (52 if rank == 0 else 140)
            node = f"token-{t}-expert-{expert}"
            drawing.box(720, center_y - 32, 320, 64, "orange", node)
            drawing.text(880, center_y - 4, f"E{expert}(x_{t}) = {expert}x_{t}", "label", size=18)
            drawing.text(880, center_y + 22, "weight = " + number(entry["weight"]), "value",
                         ident=f"moe-weight-{t}-{expert}", size=18)
            drawing.arrow([(620, top + 94), (666, top + 94), (666, center_y), (720, center_y)], f"router-{t}", node)
            drawing.arrow([(1040, center_y), (1072, center_y), (1072, top + 94), (1110, top + 94)], node, f"output-{t}")
        first, second = token["chosen"]
        drawing.text(1309, top + 66, f"y_{t} = {number(first['weight'])} E{first['expert']}(x_{t})", "small")
        drawing.text(1309, top + 93, f"+ {number(second['weight'])} E{second['expert']}(x_{t})", "small")
        drawing.text(1309, top + 128, f"= {number(token['gain'])} x_{t}", "value", ident=f"moe-output-{t}", size=21)
        selected_names = ", ".join("E" + str(entry["expert"]) for entry in token["chosen"])
        drawing.text(420, top + 178, f"Chosen: {selected_names}; unselected experts receive zero weight.", "tiny")
    drawing.text(48, 1084, "Weights and output coefficients are rounded to 3 decimals; the weighted sums use unrounded weights.", "small", "start")
    drawing.text(48, 1116, "Token positions stay separate. Capacity limits, dispatch kernels and balancing losses are outside this example.", "small", "start")
    return drawing.finish()


def scoring_svg(records):
    drawing = SVG(1600, 1000, "Four Q-Key scoring choices followed by one masked attention pipeline",
                  "Each scoring alternative takes only a query and a key. Choose one alternative to construct scores, then apply mask, row softmax, multiply the resulting weights by Value V, and return the output. Value V enters only the weighted-value stage. Cosine requires nonzero vectors; individual query and key vectors are columns.", records)
    drawing.text(48, 48, "Attention scores: choose one rule, then share one pipeline", "title", "start")
    drawing.text(48, 82, "Column vectors q_i, k_j ∈ R^Dh · batch tensors Q and K store those vectors as rows", "subtitle", "start")
    names = ("Additive", "Dot product", "Scaled dot product", "Cosine")
    for column, (method, title) in enumerate(zip(records["methods"], names)):
        left, center = 48 + column * 388, 218 + column * 388
        drawing.box(left, 128, 340, 300, "purple", "score-" + method, fill="#ffffff")
        drawing.text(center, 161, title, "heading")
        drawing.box(left + 32, 184, 110, 48, "blue")
        drawing.box(left + 198, 184, 110, 48, "purple")
        drawing.text(left + 87, 215, "q_i", "value", size=20)
        drawing.text(left + 253, 215, "k_j", "value", size=20)
        drawing.arrow([(left + 87, 232), (left + 87, 248), (center - 50, 248), (center - 50, 266)], "Q", "score-" + method)
        drawing.arrow([(left + 253, 232), (left + 253, 248), (center + 50, 248), (center + 50, 266)], "K", "score-" + method)
        drawing.box(left + 20, 266, 300, 112, "purple")
        if method == "additive":
            drawing.text(center, 295, "v_aᵀ tanh(", "formula", size=21)
            drawing.text(center, 326, "W_q q_i + W_k k_j + b)", "formula", size=19)
            drawing.text(center, 360, "e_ij ∈ R", "small")
            drawing.text(center, 405, "v_a is a learned scoring vector", "tiny")
        elif method == "dot":
            drawing.text(center, 315, "q_iᵀ k_j", "formula")
            drawing.text(center, 360, "e_ij ∈ R", "small")
        elif method == "scaled":
            drawing.text(center, 315, "q_iᵀ k_j / √(Dh)", "formula")
            drawing.text(center, 360, "e_ij ∈ R", "small")
        else:
            drawing.text(center, 299, "q_iᵀ k_j", "formula", size=21)
            drawing.items.append(f'<path d="M{left+60},311 H{left+280}" stroke="#536174" stroke-width="1.4"/>')
            drawing.text(center, 338, "||q_i||₂ ||k_j||₂", "formula", size=18)
            drawing.text(center, 365, "e_ij ∈ R", "tiny")
            drawing.text(center, 405, "q_i and k_j must both be nonzero", "tiny")
        endpoint = [650, 750, 850, 950][column]
        bend_y = 470 if column in (0, 3) else 452
        drawing.arrow([(center, 428), (center, bend_y), (endpoint, bend_y), (endpoint, 492)], "score-" + method, "choose", dashed=True)
    drawing.box(590, 492, 420, 76, "purple", "choose")
    drawing.text(800, 523, "Choose ONE scoring rule", "heading")
    drawing.text(800, 551, "One alternative supplies the score matrix", "small")
    drawing.arrow([(800, 568), (800, 625), (174, 625), (174, 720)], "choose", "scores")
    blocks = [(48, "Scores S", "scores", "purple"), (361, "Apply Mask", "mask", "orange"),
              (674, "row Softmax", "softmax", "green"), (987, "Weighted V", "weighted-v", "orange"),
              (1300, "Output", "output", "green")]
    for left, title, ident, color in blocks:
        drawing.box(left, 720, 252, 110, color, ident)
        drawing.text(left + 126, 754, title, "heading")
    drawing.text(174, 790, shape(records["scores"]), "value", size=20)
    drawing.text(174, 816, "chosen scoring rule", "tiny")
    drawing.text(487, 789, "S + M", "formula")
    drawing.text(487, 816, shape(records["scores"]), "small")
    drawing.text(800, 788, "A = softmax(S + M)", "small")
    drawing.text(800, 816, "normalize over Key axis Tk", "tiny")
    drawing.text(1113, 789, "A @ V", "formula")
    drawing.text(1113, 816, shape(records["output"]), "small")
    drawing.text(1426, 791, shape(records["output"]), "value", size=20)
    drawing.text(1426, 816, "one vector per query", "tiny")
    for (left, _, source, _), (next_left, _, target, _) in zip(blocks, blocks[1:]):
        drawing.arrow([(left + 252, 775), (next_left, 775)], source, target)
    drawing.box(987, 900, 252, 72, "orange", "value-input")
    drawing.text(1113, 930, "Value V", "label")
    drawing.text(1113, 957, shape(records["V"]), "value", size=19)
    drawing.arrow([(1113, 900), (1113, 830)], "V", "weighted-v")
    drawing.text(48, 901, "Additive: W_q,W_k ∈ R^(A×Dh); v_a,b ∈ R^A.", "small", "start")
    drawing.text(48, 932, "q_i and k_j are columns. The article’s additive example uses b = 0.", "small", "start")
    drawing.text(48, 963, "Nonempty visible-key rows shown; mask choice and head sharing are separate choices.", "small", "start")
    return drawing.finish()


def check(rows, records, causal, moe, scoring, figures):
    # Direct, unshifted Decimal exponentials provide an independent numerical
    # oracle; the renderer uses shifted binary64 exp and fsum instead.
    expected_probabilities = [("0.665", "0.245", "0.090"), ("0.731", "0.269", "0.000")]
    with localcontext() as context:
        context.prec = 70
        for index, scores in enumerate(((2, 1, 0), (2, 1, None))):
            terms = [Decimal(0) if x is None else Decimal(x).exp() for x in scores]
            probabilities = [x / sum(terms) for x in terms]
            row = rows[index]
            assert all(abs(Decimal(str(actual)) - reference) < Decimal("3e-16")
                       for actual, reference in zip(row["probabilities"], probabilities))
            assert tuple(number(x) for x in row["probabilities"]) == expected_probabilities[index]
            assert abs(sum(probabilities) - 1) < Decimal("1e-68")
            assert abs(math.fsum(row["probabilities"]) - 1) < 3e-16
            assert sum(Decimal(x) for x in expected_probabilities[index]) == Decimal("1.000")
            if index == 1:
                assert row["exponents"][2] == row["probabilities"][2] == 0
            shifted_terms = [x / Decimal(2).exp() for x in terms]
            assert tuple(number(x) for x in row["exponents"]) == tuple(f"{x:.3f}" for x in shifted_terms)
            assert number(row["total"]) == f"{sum(shifted_terms):.3f}"
    c, h, dh = records["C"], records["H"], records["Dh"]
    assert (c, h, dh) == (48, 8, 6) and c == h * dh
    # Infer linear/reshape/transpose/matmul/concat shapes independently of
    # drawing coordinates or labels. B and T remain symbolic and distinct.
    b, t, input_features = records["input"]
    out_features, weight_input = records["weight"]
    assert input_features == weight_input and records["bias"] == (out_features,)
    assert records["projected"] == (b, t, out_features)
    assert records["reshaped"] == (b, t, h, dh)
    assert tuple(records["reshaped"][i] for i in records["permutation"]) == records["split"]
    assert records["selected"] == (records["split"][0], records["split"][2], records["split"][3])
    def matrix_product(left, right):
        assert left[:-2] == right[:-2], "Batch axes must agree"
        assert left[-1] == right[-2], "Contracted dimensions must agree"
        return left[:-2] + (left[-2], right[-1])
    key_transpose = records["selected"][:-2] + (records["selected"][-1], records["selected"][-2])
    assert matrix_product(records["selected"], key_transpose) == records["scores"] == (b, t, t)
    assert matrix_product(records["scores"], records["selected"]) == records["head_output"] == (b, t, dh)
    assert records["concat_axis"] == -1 and len(records["concat"]) == h
    feature_records = [entry for entry in records["concat"]]
    assert [entry["head"] for entry in feature_records] == list(range(1, 9))
    assert all(entry["shape"] == (b, t, dh) for entry in feature_records)
    slots = [(column, entry["head"]) for entry in feature_records
             for column in range(entry["start"], entry["stop"])]
    assert [column for column, _ in slots] == list(range(c))
    assert all(head == column // dh + 1 for column, head in slots)
    assert records["merged"] == records["output"] == (b, t, c)
    # Independent direct float exponentials check the Decimal masked/shifted
    # computation. These fixed small scores do not require float stabilization.
    future_entries = 0
    with localcontext() as context:
        context.prec = 70
        assert len(causal["scores"]) == len(causal["weights"]) == 5
        for i, row in enumerate(causal["scores"]):
            assert len(row) == 5
            direct = [math.exp(float(value)) if j <= i else 0.0 for j, value in enumerate(row)]
            total = math.fsum(direct)
            assert abs(sum(causal["weights"][i]) - 1) < Decimal("1e-68")
            for j, probability in enumerate(causal["weights"][i]):
                assert abs(float(probability) - direct[j] / total) < 4e-16
                assert causal["scores"][i][j] == Decimal(CAUSAL_SCORES[i][j])
                if j > i:
                    future_entries += 1
                    assert probability == 0 and causal["masked"][i][j] is None
                else:
                    assert probability > 0 and causal["masked"][i][j] == row[j]
        assert tuple(number(p) for p in causal["weights"][1]) == ("0.130", "0.870", "0.000", "0.000", "0.000")
    assert future_entries == 10
    with localcontext() as context:
        context.prec = 70
        assert moe["top_k"] == 2 and len(moe["tokens"]) == 4
        expected_gains = ("1.269", "2.538", "2.762", "3.881")
        for token in moe["tokens"]:
            t = token["token"]
            chosen = token["chosen"]
            selected = {entry["expert"] - 1 for entry in chosen}
            assert len(chosen) == len(selected) == 2
            unselected = set(range(4)) - selected
            # This threshold condition independently characterizes Top-2 for
            # the constructed rows, which have no ties at the selection edge.
            assert min(token["logits"][j] for j in selected) > max(token["logits"][j] for j in unselected)
            routed = tuple(sum(x * weight for x, weight in zip(token["x"], row))
                           for row in moe["router_weight"])
            assert routed == token["logits"]
            direct = {j: math.exp(float(token["logits"][j])) for j in selected}
            denominator = math.fsum(direct.values())
            assert abs(sum(entry["weight"] for entry in chosen) - 1) < Decimal("1e-68")
            for entry in chosen:
                j = entry["expert"] - 1
                assert entry["gain"] == moe["expert_gains"][j] == j + 1
                assert abs(float(entry["weight"]) - direct[j] / denominator) < 4e-16
            float_gain = math.fsum((j + 1) * direct[j] / denominator for j in selected)
            assert abs(float(token["gain"]) - float_gain) < 2e-15
            assert number(token["gain"]) == expected_gains[t - 1]
            assert token["output"][t - 1] == token["gain"]
            assert all(value == 0 for d, value in enumerate(token["output"]) if d != t - 1)
    key_batch_transpose = scoring["K"][:-2] + (scoring["K"][-1], scoring["K"][-2])
    assert matrix_product(scoring["Q"], key_batch_transpose) == scoring["scores"]
    assert matrix_product(scoring["weights"], scoring["V"]) == scoring["output"]
    columns = scoring["column_vectors"]
    assert matrix_product(columns["W_q"], columns["q_i"]) == columns["b"]
    assert matrix_product(columns["W_k"], columns["k_j"]) == columns["b"]
    assert matrix_product(tuple(reversed(columns["v_a"])), columns["b"]) == (1, 1)
    assert scoring["cosine_requires_nonzero_vectors"]
    for name, source in figures.items():
        root = ET.fromstring(source)
        ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
        assert len(ids) == len(set(ids)), name
        assert root.find("s:title", NS) is not None and root.find("s:desc", NS) is not None
        assert root.attrib["viewBox"] == "0 0 " + root.attrib["width"] + " " + root.attrib["height"]
        if name == "multi-head-shapes.svg":
            edges = {(node.attrib["data-from"], node.attrib["data-to"])
                     for node in root.iter() if "data-from" in node.attrib}
            assert {src for src, dst in edges if dst == "attention"} == {"q-i", "k-i", "v-i"}
            for letter in "qkv":
                assert ("x", letter + "-project") in edges
                assert (letter + "-project", letter + "-split") in edges
                assert (letter + "-split", letter + "-i") in edges
            assert ("concat", "output-linear") in edges
            assert len([ident for ident in ids if ident.startswith("head-output-")]) == 8
        elif name == "softmax-row-calculation.svg":
            sums = [node.text for node in root.findall("s:text", NS) if node.attrib.get("id", "").startswith("row-")]
            assert sums == ["0.665 + 0.245 + 0.090 = 1.000", "0.731 + 0.269 + 0.000 = 1.000"]
        elif name == "causal-mask-calculation.svg":
            cells = {node.attrib.get("id"): node.text for node in root.findall("s:text", NS)}
            for field in ("scores", "masked", "weights"):
                for i, row in enumerate(causal[field], 1):
                    for j, value in enumerate(row, 1):
                        assert cells[f"causal-{field}-{i}-{j}"] == number(value)
        elif name == "moe-token-routing.svg":
            edges = {(node.attrib["data-from"], node.attrib["data-to"])
                     for node in root.iter() if "data-from" in node.attrib}
            cells = {node.attrib.get("id"): node.text for node in root.findall("s:text", NS)}
            for token in moe["tokens"]:
                t = token["token"]
                expert_nodes = {f"token-{t}-expert-{entry['expert']}" for entry in token["chosen"]}
                assert {dst for src, dst in edges if src == f"router-{t}"} == expert_nodes
                assert {src for src, dst in edges if dst == f"output-{t}"} == expert_nodes
                assert (f"token-{t}", f"router-{t}") in edges
                for entry in token["chosen"]:
                    assert cells[f"moe-weight-{t}-{entry['expert']}"] == "weight = " + number(entry["weight"])
                assert cells[f"moe-output-{t}"] == f"= {number(token['gain'])} x_{t}"
            assert len(edges) == 4 * (1 + 2 + 2)
        elif name == "attention-scoring-pipeline.svg":
            edges = {(node.attrib["data-from"], node.attrib["data-to"])
                     for node in root.iter() if "data-from" in node.attrib}
            nodes = {node for edge in edges for node in edge}
            successors = {node: {dst for src, dst in edges if src == node} for node in nodes}
            predecessors = {node: {src for src, dst in edges if dst == node} for node in nodes}
            for method in scoring["methods"]:
                node = "score-" + method
                assert predecessors[node] == {"Q", "K"}
                assert successors[node] == {"choose"}
            assert predecessors["choose"] == {"score-" + method for method in scoring["methods"]}
            assert successors["choose"] == {"scores"}
            for start, end in (("scores", "mask"), ("mask", "softmax"),
                               ("softmax", "weighted-v"), ("weighted-v", "output")):
                assert successors[start] == {end}
            assert predecessors["mask"] == {"scores"}
            assert predecessors["softmax"] == {"mask"}
            assert predecessors["weighted-v"] == {"softmax", "V"}
            assert predecessors["output"] == {"weighted-v"}
            assert successors["V"] == {"weighted-v"}

            def descendants(node, trail=frozenset()):
                assert node not in trail, "Scoring graph must be acyclic"
                return set().union(*(descendants(child, trail | {node}) | {child}
                                     for child in successors[node]))
            for node in nodes:
                descendants(node)
            assert descendants("V") == {"weighted-v", "output"}
            assert descendants("scores") == {"mask", "softmax", "weighted-v", "output"}
        else:
            raise AssertionError("Unchecked figure: " + name)
    return {"decimal_softmax_rows": 2, "precision_digits": 70,
            "mha_dimensions": {"C": c, "H": h, "Dh": dh},
            "concat_features": len(slots), "all_three_inputs_reach_each_head_template": True,
            "causal_rows": 5, "causal_exact_zero_future_entries": future_entries,
            "moe_tokens": 4, "moe_chosen_edges_per_token": 2,
            "moe_selected_softmax_and_separate_outputs": "passed",
            "scoring_dag_mask_before_softmax_and_value_only_at_weighted_sum": "passed",
            "svg_xml_and_labels": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, help="directory to receive the SVG files")
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error("use --check, --output DIRECTORY, or both")
    rows, records, causal = softmax_records(), mha_records(), causal_records()
    moe, scoring = moe_records(), scoring_records()
    figures = {"softmax-row-calculation.svg": softmax_svg(rows),
               "multi-head-shapes.svg": mha_svg(records),
               "causal-mask-calculation.svg": causal_svg(causal),
               "moe-token-routing.svg": moe_svg(moe),
               "attention-scoring-pipeline.svg": scoring_svg(scoring)}
    report = check(rows, records, causal, moe, scoring, figures)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, source in figures.items():
            (args.output / name).write_text(source, encoding="utf-8")
        report["output"] = str(args.output)
    if args.check:
        stale = [name for name, source in figures.items()
                 if not (ASSETS / name).exists() or (ASSETS / name).read_text() != source]
        assert not stale, "Rebuild stale generated assets: " + ", ".join(stale)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
