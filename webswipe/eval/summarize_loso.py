# leave-one-subject-out summary over all D1 subjects: results/*_s<id>.json -> markdown tables
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

DECODERS = ["SHARK2+smooth@50Hz", "clean@50Hz", "aug@50Hz"]


def load(pattern, exclude=()):
    out = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"_s(\d+)\.json$", p)
        if m and int(m.group(1)) not in exclude:  # also skips older files such as robustness_d1_shark2_rs50.json
            out[int(m.group(1))] = json.load(open(p))
    return out


def ms(v):
    v = 100 * np.asarray(v)
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results/loso_summary.md")
    ap.add_argument("--exclude", nargs="*", type=int, default=[], help="held-out subjects to leave out")
    a = ap.parse_args()
    rob = load(f"{a.dir}/robustness_d1_s*.json", a.exclude)
    lm = load(f"{a.dir}/lm_prior_d1_s*.json", a.exclude)
    subs = sorted(rob)
    conds = list(rob[subs[0]]["conditions"])
    md = [f"# D1 leave-one-subject-out summary ({len(subs)} held-out subjects: {', '.join(map(str, subs))})",
          "", "Top-1 word accuracy (%), mean ± sd over held-out subjects; every decoder on traces resampled to 50 Hz.",
          "`aug > clean` = held-out subjects where the augmented model beats the clean one.", "",
          "| Condition | " + " | ".join(DECODERS) + " | aug − clean | aug > clean |",
          "|---|" + "---|" * (len(DECODERS) + 2)]
    for c in conds:
        top1 = {d: [rob[s]["conditions"][c][d][0] for s in subs] for d in DECODERS}
        diff = np.subtract(top1["aug@50Hz"], top1["clean@50Hz"])
        md.append(f"| {c} | " + " | ".join(ms(top1[d]) for d in DECODERS)
                  + f" | {100 * diff.mean():+.1f} | {(diff > 0).sum()}/{len(subs)} |")

    md += ["", "Per held-out subject, webcam preset (12 fps + 0.15 + 100 ms), top-1 (%):", "",
           "| Subject | n | " + " | ".join(DECODERS) + " |", "|---|---|" + "---|" * len(DECODERS)]
    wc = [c for c in conds if c.startswith("webcam")][0]
    for s in subs:
        md.append(f"| {s} | {rob[s]['n_test']} | "
                  + " | ".join(f"{100 * rob[s]['conditions'][wc][d][0]:.1f}" for d in DECODERS) + " |")

    if lm:
        ls = sorted(lm)
        lams = lm[ls[0]]["lambdas"]
        md += ["", f"Word-frequency prior (score = log-lik + λ·log p(word)), top-1 (%), mean over {len(ls)} subjects.",
               "SHARK2's λ is tuned per held-out subject on the other subjects "
               f"(chosen: {', '.join(str(lm[s]['shark2_tuned_lambda']) for s in ls)}).", "",
               "| Condition | Decoder | " + " | ".join(f"λ={l:g}" for l in lams) + " | SHARK2 tuned λ |",
               "|---|---|" + "---|" * (len(lams) + 1)]
        for c in lm[ls[0]]["conditions"]:
            for d in lm[ls[0]]["conditions"][c]:
                cells = [ms([lm[s]["conditions"][c][d][str(float(l))][0] for s in ls]) for l in lams]
                tuned = ms([lm[s]["conditions"][c][d][str(float(lm[s]["shark2_tuned_lambda"]))][0] for s in ls]) \
                    if d.startswith("SHARK2") else ""
                md.append(f"| {c} | {d} | " + " | ".join(cells) + f" | {tuned} |")

    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
