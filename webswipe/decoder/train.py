# train / fine-tune the ported Shen decoder
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .decode import Lexicon, greedy, lexicon_accuracy
from .features import encode_word
from .model import ShenDecoder, ctc_loss


def load(paths):
    X, W, S = [], [], []
    for i, p in enumerate(paths):
        d = np.load(p)
        X.append(d["X"]); W.append(d["words"])
        # prefix subject ids with the file index so different datasets never collide
        S.append(np.array([f"{i}:{s}" for s in d["subject"]]))
    return np.concatenate(X), np.concatenate(W), np.concatenate(S)


def split(W, S, how, test_subject, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(W))
    if how == "subject":
        test = S[idx] == test_subject
        return idx[~test], idx[test]
    n_val = int(len(idx) * val_frac)
    return idx[n_val:], idx[:n_val]


def batches(idx, bs, shuffle, rng):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def targets(words):
    enc = [encode_word(w) for w in words]
    lens = torch.tensor([len(e) for e in enc])
    t = torch.zeros(len(enc), int(lens.max()), dtype=torch.long)
    for i, e in enumerate(enc):
        t[i, :len(e)] = torch.tensor(e)
    return t, lens


@torch.no_grad()
def evaluate(model, X, W, idx, lex, device, bs=256, lexicon=True):
    model.eval()
    loss, logits_all = 0.0, []
    for b in batches(idx, bs, False, None):
        lg = model(torch.from_numpy(X[b]).to(device))
        t, tl = targets(W[b])
        loss += ctc_loss(lg, t.to(device), tl.to(device)).item() * len(b)
        logits_all.append(lg.cpu())
    logits = torch.cat(logits_all)
    words = W[idx]
    res = {"val_loss": loss / len(idx),
           "greedy_acc": float(np.mean([p == w for p, w in zip(greedy(logits), words)]))}
    if lexicon:
        res["top1"], res["top4"] = lexicon_accuracy(logits, words, lex)
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/shen_features.npz"])
    ap.add_argument("--split", choices=["random", "subject"], default="random")
    ap.add_argument("--test-subject", default="0:0", help="file_index:subject_id, for --split subject")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", help="checkpoint to fine-tune from")
    ap.add_argument("--freeze", choices=["none", "lstm1", "lstm"], default="none")
    ap.add_argument("--max-train", type=int, default=0, help="use only this many training samples")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--out", default="checkpoints/shen_d1.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    X, W, S = load(a.data)
    tr, va = split(W, S, a.split, a.test_subject, a.val_frac, a.seed)
    if a.max_train:
        tr = tr[:a.max_train]
    lex = Lexicon(sorted(set(W)))
    print(f"train {len(tr)}  val {len(va)}  lexicon {len(lex.words)}  device {a.device}")

    model = ShenDecoder().to(a.device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=a.device)["model"])
    if a.freeze != "none":
        frozen = model.lstms[:1] if a.freeze == "lstm1" else model.lstms
        for p in frozen.parameters():
            p.requires_grad = False
    opt = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0, tot = time.time(), 0.0
        for b in batches(tr, a.batch, True, rng):
            t, tl = targets(W[b])
            loss = ctc_loss(model(torch.from_numpy(X[b]).to(a.device)), t.to(a.device), tl.to(a.device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        row = {"epoch": ep, "train_loss": tot / len(tr), "sec": round(time.time() - t0, 1)}
        if ep % a.eval_every == 0 or ep == a.epochs:
            row.update(evaluate(model, X, W, va, lex, a.device))
            if row["top1"] > best:
                best = row["top1"]
                torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep, "metrics": row}, a.out)
        hist.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).with_suffix(".json").write_text(json.dumps({"args": vars(a), "history": hist}, indent=1))
    print(f"best val top-1 {best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
