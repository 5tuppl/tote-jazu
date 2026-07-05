"""Deterministic inverse lookup-table baseline (honest comparison for RLVR)."""
import json, random, torch
import tote_grpo as T
from collections import Counter, defaultdict

SEED = 0
random.seed(SEED); torch.manual_seed(SEED)
pairs = T.make_corpus(); tr, te = T.split(pairs)

# Map each Arabic glyph to its most frequent Cyrillic pre-image in TRAIN.
co = defaultdict(Counter)
for ar, cy in tr:
    for a, c in zip(ar, cy):
        co[a][c] += 1
inv = {a: cnt.most_common(1)[0][0] for a, cnt in co.items()}
def lut(ar): return "".join(inv.get(a, a) for a in ar)

ex = 0; cers = []
for ar, cy in te:
    p = lut(ar); cers.append(T.cer(p, cy)); ex += (p == cy)
res = {"exact": ex / len(te), "cer": sum(cers) / len(cers)}
json.dump({"lut": res}, open("m_lut.json", "w"))
print(f"LOOKUP TABLE  test exact={res['exact']:.3f} cer={res['cer']:.3f}")
