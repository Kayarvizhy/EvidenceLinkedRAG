"""make_figure.py -- Figure 1, architecture of the revised method.  Output: results/figure1.jpg"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 7.645, 4.385                                   # same size as the figure it replaces
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 57.4)
ax.axis("off")
INK, MUTE = "#1f2a44", "#5b6472"
COL = {"blue": ("#eef4fd", "#4a7fd0"), "orange": ("#fff5e8", "#e0922a"), "green": ("#eef8ee", "#4f9a55"),
       "purple": ("#f3effb", "#7a5bc0"), "red": ("#fdf0ef", "#d0524a"), "grey": ("#f4f5f7", "#8a93a3")}


def panel(x, y, w, h, c, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.9", fc=COL[c][0], ec=COL[c][1], lw=0.9))
    ax.text(x + w / 2, y + h - 1.9, title, ha="center", va="center", fontsize=6.3, fontweight="bold", color=INK)


def box(x, y, w, h, text, c, fs=5.3, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.5", fc="white", ec=COL[c][1], lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x1, y1, x2, y2, c=INK, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=c,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))


TOP, PH = 21.5, 34.5
# 0 input
panel(0.8, TOP, 9.4, PH, "grey", "Input")
box(1.6, TOP + 20.5, 7.8, 8.5, "Question\nQ", "grey", 5.6, True)
box(1.6, TOP + 3, 7.8, 15, "Example\n\nWhat position was\nheld by the woman\nwho played Corliss\nArcher in Kiss\nand Tell?", "grey", 4.5)
# 1 query understanding
panel(11.6, TOP, 12.2, PH, "blue", "1. Query\nunderstanding")
box(12.5, TOP + 18, 10.4, 9.5, "Intent\nfactual / analytical /\nprocedural", "blue", 5.0)
box(12.5, TOP + 3, 10.4, 12.5, "Routes to\ngenerator and prompt;\nsecond hop is skipped\nfor procedural\nquestions", "blue", 4.7)
arrow(17.7, TOP + 18, 17.7, TOP + 15.5)
# 2 first-stage retrieval
panel(25.2, TOP, 17.4, PH, "orange", "2. Hybrid retrieval:\nparagraphs + sentences")
box(26.1, TOP + 21.2, 7.6, 6.3, "BM25\nparagraphs", "orange", 4.9)
box(34.1, TOP + 21.2, 7.6, 6.3, "Dense\nparagraphs", "orange", 4.9)
box(26.1, TOP + 13.2, 7.6, 6.3, "BM25\nsentences", "orange", 4.9)
box(34.1, TOP + 13.2, 7.6, 6.3, "Dense\nsentences", "orange", 4.9)
box(26.1, TOP + 8.2, 15.6, 3.6, "sentence  →  parent paragraph", "orange", 4.7)
box(26.1, TOP + 2.2, 15.6, 4.6, "Reciprocal rank fusion\ncandidate pool (50)", "orange", 4.9, True)
arrow(33.9, TOP + 13.2, 33.9, TOP + 11.8)
arrow(33.9, TOP + 8.2, 33.9, TOP + 6.8)
# 3 rerank + second hop
panel(44.0, TOP, 18.4, PH, "green", "3. Reranking and\nsecond hop")
box(44.9, TOP + 21.6, 16.6, 6.0, "Cross-encoder\n$s_1$ = CE(Q, d)", "green", 5.0)
box(44.9, TOP + 14.3, 16.6, 5.8, "Title links of the\ntop 3 paragraphs", "green", 5.0)
box(44.9, TOP + 7.2, 16.6, 5.6, "Path score\n$s_2$ = CE(Q, b ⊕ d)", "green", 5.0)
box(44.9, TOP + 1.6, 16.6, 4.4, "Context: 4 first-hop +\n1 second-hop paragraph", "green", 4.8, True)
for a, b in ((21.6, 20.1), (14.3, 12.8), (7.2, 6.0)):
    arrow(53.2, TOP + a, 53.2, TOP + b)
# 4 generation
panel(63.8, TOP, 13.0, PH, "purple", "4. Generation")
box(64.6, TOP + 19.5, 11.4, 8.0, "LLM generator\nP(A | Q, D)", "purple", 5.0)
box(64.6, TOP + 10.5, 11.4, 7.0, "Intent-specific\nprompt", "purple", 5.0)
box(64.6, TOP + 3, 11.4, 5.5, "First answer A", "purple", 5.0, True)
arrow(70.3, TOP + 10.5, 70.3, TOP + 8.5)
# 5 verification
panel(78.2, TOP, 12.6, PH, "red", "5. Answer\nverification")
box(79.0, TOP + 12.3, 11.0, 15.2, "Reject if\n• abstention\n• answer type does\n  not fit the question\n• evidence overlap\n  below 0.5", "red", 4.6)
ax.add_patch(plt.Polygon([[84.5, TOP + 10.6], [89.3, TOP + 6.4], [84.5, TOP + 2.2], [79.7, TOP + 6.4]], fc="white", ec=COL["red"][1], lw=0.7))
ax.text(84.5, TOP + 6.4, "accepted?", ha="center", va="center", fontsize=4.8, color=INK)
arrow(84.5, TOP + 12.3, 84.5, TOP + 10.6)
# 6 output
panel(92.2, TOP, 7.0, PH, "grey", "Output")
box(92.8, TOP + 3.4, 5.8, 6.0, "Final\nanswer", "grey", 5.0, True)
arrow(89.3, TOP + 6.4, 92.8, TOP + 6.4, "#2e8b57")
ax.text(91.0, TOP + 7.7, "yes", fontsize=4.6, color="#2e8b57", ha="center")
# flow arrows between panels
for x1, x2 in ((10.2, 11.6), (23.8, 25.2), (42.6, 44.0), (62.4, 63.8), (76.8, 78.2)):
    arrow(x1, TOP + PH / 2 + 3, x2, TOP + PH / 2 + 3)
# refinement
panel(52.0, 1.2, 47.2, 17.0, "red", "6. Refinement (at most once)")
box(53.0, 3.0, 14.2, 9.8, "Wider context\n8 paragraphs:\n5 first-hop +\n3 second-hop", "red", 4.7)
box(68.4, 3.0, 14.2, 9.8, "Answer-type hint\nadded to the prompt\n(yes/no, date,\nnumber, name)", "red", 4.7)
box(83.8, 3.0, 14.4, 9.8, "Regenerate;\nkeep the new answer\nunless it is an\nabstention", "red", 4.7)
arrow(67.2, 7.9, 68.4, 7.9)
arrow(82.6, 7.9, 83.8, 7.9)
arrow(84.5, TOP + 2.2, 84.5, 18.2, COL["red"][1])
ax.text(85.6, 19.9, "no", fontsize=4.6, color=COL["red"][1])
arrow(95.7, 12.8, 95.7, TOP + 3.4, COL["red"][1])
# offline
panel(0.8, 1.2, 49.6, 17.0, "grey", "Knowledge base construction (offline)")
box(1.8, 3.0, 10.6, 9.8, "Documents\nwith titles", "grey", 4.9)
box(13.6, 3.0, 11.2, 9.8, "Segmentation\nparagraphs and\nsentences", "grey", 4.8)
box(26.0, 3.0, 11.2, 9.8, "BM25 and dense\nindexes: paragraphs\nand sentences", "grey", 4.8)
box(38.4, 3.0, 11.0, 9.8, "Title index\nfor second-hop\nlinks", "grey", 4.8)
for x in (12.4, 24.8, 37.2):
    arrow(x, 7.9, x + 1.2, 7.9)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
fig.savefig(os.path.join(HERE, "results", "figure1.jpg"), dpi=300, facecolor="white", pil_kwargs={"quality": 95})
fig.savefig(os.path.join(HERE, "results", "figure1_preview.png"), dpi=110, facecolor="white")
print("saved")
