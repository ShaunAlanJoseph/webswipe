# figure for results/robustness_d1.json: top-1 accuracy per condition, one bar per decoder
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots in fixed order (validated: dataviz palette, light surface)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/robustness_d1.json")
    ap.add_argument("--out", default="results/robustness_d1.png")
    a = ap.parse_args()
    res = json.load(open(a.json))
    conds = list(res["conditions"])
    decoders = list(res["conditions"][conds[0]])
    n, m = len(conds), len(decoders)
    h = 0.8 / m

    fig, ax = plt.subplots(figsize=(8, 0.75 * n + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(n)[::-1]
    for j, (d, c) in enumerate(zip(decoders, COLORS)):
        v = [res["conditions"][k][d][0] for k in conds]
        yy = y + 0.4 - h * (j + 0.5)
        ax.barh(yy, v, height=h - 0.02, color=c, label=d)  # small gap between adjacent bars
        for yi, vi in zip(yy, v):
            ax.text(vi + 0.008, yi, f"{vi * 100:.0f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, conds, fontsize=9, color=INK)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Top-1 word accuracy (held-out subject, 1,996-word lexicon, no LM)", color=INK)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("D1 under webcam-like degradation", color=INK, loc="left")
    plt.tight_layout()
    plt.savefig(a.out, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
