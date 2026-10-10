"""Resolve reproducible local decoder defaults before camera startup."""
from pathlib import Path


def configure_decoder(args, root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    if args.lexicon is None:
        wide = root / 'data/lexicon_10k.txt'
        args.lexicon = str(wide if wide.is_file() else root / 'data/shen_features.npz')
    if args.mode in ('type', 'guided'):
        if args.decoder is None:
            if not args.ckpt:
                for candidate in [root / 'checkpoints/d1_loso0_aug_last.pt',
                                  root / '../webswipe-eval/kaggle_build/out/train-0/out/checkpoints/d1_loso0_aug_last.pt']:
                    if candidate.is_file():
                        args.ckpt = str(candidate.resolve())
                        break
            args.decoder = 'neural' if args.ckpt else 'shark2'
        if args.decoder == 'neural' and not args.ckpt:
            raise ValueError('Neural decoder needs --ckpt PATH')
    if args.resample_hz is None:
        args.resample_hz = 50.0
    if args.prior_weight is None:
        args.prior_weight = 0.0 if args.decoder == 'neural' else 0.25
    return args
