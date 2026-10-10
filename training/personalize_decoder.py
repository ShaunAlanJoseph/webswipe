"""Personalize the BiLSTM-CTC word decoder to one user (produces decoder_personalized_20261010).

Fine-tunes the D1 leave-one-subject-out decoder (subject 0 held out) on the
user's mid-air traces from decoder_adaptation_payload.json, mixed each epoch
with 512 random Shen et al. D1 samples from the other subjects. Each air trace
is used 4 times per epoch (3 jittered copies). RMSprop, lr 1e-5, 10 epochs.

Epochs 2, 5 and 10 are scored on the development traces (lexicon top-1/top-4
with endpoint cost), the 5 supplementary older-session traces, and greedy CTC
accuracy on 512 fixed D1 subject-0 samples. A checkpoint is kept only if it
gains >= 2 top-1 on the 45 development words, loses no top-4 overall or per
posture, does not drop on the supplementary traces, and loses at most 5 of the
512 D1 greedy words. The best passing epoch is saved as decoder_candidate.pt.

Needs a CUDA GPU.

Example:
    python training/personalize_decoder.py --init models/d1_loso0_aug_last.pt \
        --payload data/decoder_adaptation/decoder_adaptation_payload.json \
        --features data/shen_features.npz --lexicon data/lexicon_10k.txt
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--init', type=Path, default=REPO / 'models/d1_loso0_aug_last.pt',
                   help='starting decoder checkpoint (dict with a "model" state dict)')
    p.add_argument('--payload', type=Path, default=REPO / 'data/decoder_adaptation/decoder_adaptation_payload.json',
                   help='train / development / supplementary_validation traces')
    p.add_argument('--features', type=Path, default=REPO / 'data/shen_features.npz',
                   help='Shen et al. D1 features (X, words, subject)')
    p.add_argument('--lexicon', type=Path, default=REPO / 'data/lexicon_10k.txt')
    p.add_argument('--out', type=Path, default=REPO / 'runs/decoder_personalized')
    return p.parse_args()


def main():
    args = parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def progress(**row):
        row['elapsed_s'] = round(time.time() - started, 2)
        print(json.dumps(row), flush=True)

    try:
        import numpy as np, torch
        from webswipe.decoder.model import ShenDecoder, ctc_loss
        from webswipe.decoder.features import featurize, resample_time, Keyboard, endpoint_cost
        from webswipe.decoder.train import targets
        from webswipe.decoder.decode import Lexicon, greedy
        torch.set_num_threads(2)
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        assert torch.cuda.is_available(), 'GPU required; do not silently run training on CPU'
        device = 'cuda'
        payload_path = args.payload
        payload = json.loads(payload_path.read_text())
        d = np.load(args.features)
        X, W, S = d['X'], d['words'], d['subject'].astype(str)
        replay_ids = np.flatnonzero(S != '0')
        held_ids = rng.choice(np.flatnonzero(S == '0'), 512, replace=False)
        assert len(replay_ids) > 512
        words = sorted(set(w.strip().lower() for w in args.lexicon.read_text().splitlines() if w.strip()))
        lex = Lexicon(words)

        def features(row, augment=False):
            tr = np.asarray(row['trace'], float).copy()
            pitch = row['pitch']
            kb = Keyboard.from_pitch(pitch, pitch)
            if augment:
                tr[:, :2] = tr[:, :2] * rng.uniform(.98, 1.02) + rng.normal(0, .06 * pitch, (1, 2))
            return featurize(resample_time(tr[:, :2], tr[:, 2], 50), kb)

        dev = payload['development']
        supp = payload['supplementary_validation']
        air = payload['train']
        devX = np.stack([features(r) for r in dev])
        suppX = np.stack([features(r) for r in supp])
        dev_cost = [torch.tensor(endpoint_cost(np.asarray(r['trace']), Keyboard.from_pitch(r['pitch'], r['pitch']), words))
                    for r in dev]
        supp_cost = [torch.tensor(endpoint_cost(np.asarray(r['trace']), Keyboard.from_pitch(r['pitch'], r['pitch']), words))
                     for r in supp]
        model = ShenDecoder().to(device)
        model.load_state_dict(torch.load(args.init, map_location=device)['model'])

        @torch.no_grad()
        def logits_for(feat):
            return torch.cat([model(torch.from_numpy(feat[i:i + 32]).to(device)).cpu() for i in range(0, len(feat), 32)])

        @torch.no_grad()
        def evaluate(epoch):
            model.eval()
            lg = logits_for(devX)
            slg = logits_for(suppX)
            pred = []
            spred = []
            for i, r in enumerate(dev):
                idx = torch.topk(lex.scores(lg[i], 0) - dev_cost[i], 4).indices.tolist()
                pred.append(dict(id=r['id'], target=r['target'], condition=r['condition'], words=[words[j] for j in idx]))
            for i, r in enumerate(supp):
                idx = torch.topk(lex.scores(slg[i], 0) - supp_cost[i], 4).indices.tolist()
                spred.append(dict(id=r['id'], target=r['target'], words=[words[j] for j in idx]))
            generic = greedy(logits_for(X[held_ids]))
            res = {'epoch': epoch, 'postures': {},
                   'supplementary_top1': sum(r['words'][0] == r['target'] for r in spred),
                   'supplementary_top4': sum(r['target'] in r['words'] for r in spred),
                   'supplementary_n': len(spred),
                   'D1_greedy_correct': sum(a == b for a, b in zip(generic, W[held_ids])), 'D1_n': len(held_ids),
                   'predictions': pred, 'supplementary_predictions': spred}
            for c, n in [('sitting', 20), ('standing', 25), ('combined', 45)]:
                rows = [r for r in pred if c == 'combined' or r['condition'] == c]
                res['postures'][c] = {'top1': sum(r['words'][0] == r['target'] for r in rows),
                                      'top4': sum(r['target'] in r['words'] for r in rows), 'n': n}
            progress(stage='evaluation', epoch=epoch, metrics={k: v for k, v in res.items() if 'predictions' not in k})
            model.train()
            return res

        progress(stage='baseline_evaluation', air_n=len(air))
        history = [evaluate(0)]
        baseline = history[0]
        # The starting decoder must reproduce the frozen baseline (35/45 top-1, 41/45 top-4).
        assert baseline['postures']['combined'] == {'top1': 35, 'top4': 41, 'n': 45}, baseline['postures']
        opt = torch.optim.RMSprop(model.parameters(), lr=1e-5)
        chosen = None
        for epoch in range(1, 11):
            feats = []
            labels = []
            for repeat in range(4):
                for row in air:
                    feats.append(features(row, augment=repeat > 0))
                    labels.append(row['target'])
            ids = rng.choice(replay_ids, 512, replace=False)
            xb = np.concatenate([np.stack(feats), X[ids]])
            wb = np.asarray(labels + W[ids].tolist())
            order = rng.permutation(len(wb))
            total = 0
            model.train()
            for start in range(0, len(order), 32):
                b = order[start:start + 32]
                target, tl = targets(wb[b])
                opt.zero_grad()
                loss = ctc_loss(model(torch.from_numpy(xb[b]).to(device)), target.to(device), tl.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                total += float(loss.detach()) * len(b)
            progress(stage='training', epoch=epoch, loss=total / len(order))
            if epoch in [2, 5, 10]:
                row = evaluate(epoch)
                history.append(row)
                m = row['postures']
                b = baseline['postures']
                accepted = (m['combined']['top1'] >= b['combined']['top1'] + 2
                            and all(m[c]['top4'] >= b[c]['top4'] for c in ['sitting', 'standing', 'combined'])
                            and row['supplementary_top1'] >= baseline['supplementary_top1']
                            and row['supplementary_top4'] >= baseline['supplementary_top4']
                            and row['D1_greedy_correct'] >= baseline['D1_greedy_correct'] - 5)
                row['passes_predeclared_gate'] = accepted
                if accepted and (chosen is None or (m['combined']['top4'], m['combined']['top1'])
                                 > (chosen['postures']['combined']['top4'], chosen['postures']['combined']['top1'])):
                    chosen = row
                    torch.save({'model': model.state_dict(), 'epoch': epoch, 'metrics': row,
                                'args': {'lr': 1e-5, 'seed': 42, 'air_train_n': len(air), 'D1_subject_0_excluded': True}},
                               OUT / 'decoder_candidate.pt')
                (OUT / 'history.json').write_text(json.dumps(history, indent=2))
        torch.save({'model': model.state_dict(), 'epoch': 10}, OUT / 'decoder_last.pt')
        (OUT / 'result.json').write_text(json.dumps({
            'finished': True, 'selected_epoch': chosen['epoch'] if chosen else None, 'passes_gate': chosen is not None,
            'elapsed_s': time.time() - started, 'payload_sha256': hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            'D1_validation_ids': held_ids.tolist(), 'training_subjects': sorted(set(S[replay_ids]))}, indent=2))
        progress(stage='complete', selected=chosen['epoch'] if chosen else None)
    except Exception as e:
        (OUT / 'failure.json').write_text(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))
        raise


if __name__ == '__main__':
    main()
