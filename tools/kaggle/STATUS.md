# Kaggle runs (Oct 4-5): status and how to resume

Kernels are private under kaggle.com/shauniiitk; each keeps its outputs on Kaggle after it finishes.
Check: `.venv/bin/kaggle kernels status shauniiitk/<slug>`; fetch: `.venv/bin/kaggle kernels output shauniiitk/<slug> -p kaggle_build/out/<slug>`.

| # | Run | Kernel slug | Status |
|---|---|---|---|
| 0 | smoke | webswipe-smoke | done |
| 1 | train subject 0 (clean + aug) | webswipe-train-0 | done; histories in `results/d1_loso0_*.json` |
| 2 | eval subject 0 (CPU) | webswipe-eval-0 | done; `results/*_s0.json`, `results/robustness_d1_s0.png` |
| 3 | Shen reproduction, random split, 200 epochs | webswipe-repro | done; 92.2 top-1 / 97.3 top-4 (last epoch), `results/shen_d1_random.json` |
| 4a | train subjects 1,2,4,5,6,7,8,9 | webswipe-train-1-2-4-5-6-7-8-9 | done Oct 5 02:53; histories in `results/d1_loso*_*.json` |
| 4b | eval those subjects, 4 CPU kernels | webswipe-eval-1-2, -4-5, -6-7, -8-9 | done Oct 5 05:13; `results/loso_summary*.md` |

Resume (from the repository root):
```
.venv/bin/python tools/kaggle/push.py train:1,2,4,5,6,7,8,9
# after it completes:
for g in 1,2 4,5 6,7 8,9; do .venv/bin/python tools/kaggle/push.py eval:$g --cpu --after train:1,2,4,5,6,7,8,9; done
```
