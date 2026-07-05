"""Stage 1 pipeline over a pairs TSV. Configurable via env:
   PAIRS (default pairs.tsv), TAG (default s1)."""
import sys, os, json, time, random, torch
import tote_grpo as T
from collections import Counter, defaultdict

SEED = 0
PAIRS = os.environ.get("PAIRS", "pairs.tsv")
TAG = os.environ.get("TAG", "s1")

def load():
    pairs = []
    for line in open(PAIRS, encoding="utf-8"):
        if "\t" in line:
            ar, cy = line.rstrip("\n").split("\t"); pairs.append((ar, cy))
    random.seed(SEED); torch.manual_seed(SEED)
    Vs = T.Vocab([p[0] for p in pairs]); Vt = T.Vocab([p[1] for p in pairs])
    tr, te = T.split(pairs)
    return pairs, Vs, Vt, tr, te

phase = sys.argv[1]
pairs, Vs, Vt, tr, te = load()

if phase == "info":
    print(f"[{TAG}] pairs={len(pairs)} train={len(tr)} test={len(te)} "
          f"src_vocab={len(Vs)} tgt_vocab={len(Vt)}")

elif phase == "lut":
    co = defaultdict(Counter)
    for ar, cy in tr:
        if len(ar) == len(cy):
            for a, c in zip(ar, cy): co[a][c] += 1
    inv = {a: cnt.most_common(1)[0][0] for a, cnt in co.items()}
    def lut(ar): return "".join(inv.get(a, a) for a in ar)
    ex = 0; cers = []
    for ar, cy in te:
        p = lut(ar); cers.append(T.cer(p, cy)); ex += (p == cy)
    res = {"exact": ex/len(te), "cer": sum(cers)/len(cers)}
    json.dump({"lut": res}, open(f"{TAG}_lut.json", "w"))
    print(f"[{TAG}] LOOKUP  exact={res['exact']:.3f} cer={res['cer']:.3f}")

elif phase in ("sft_light", "sft_full"):
    steps = 1200 if phase == "sft_light" else 3000
    random.seed(SEED); torch.manual_seed(SEED)
    m = T.Seq2Seq(len(Vs), len(Vt))
    t = time.time(); m = T.sft(m, tr, Vs, Vt, steps=steps, lr=1e-3)
    em, ce = T.evaluate(m, te, Vs, Vt)
    torch.save(m.state_dict(), f"{TAG}_{phase}.pt")
    json.dump({phase: {"exact": em, "cer": ce, "steps": steps}}, open(f"{TAG}_{phase}.json", "w"))
    print(f"[{TAG}] {phase} {time.time()-t:.0f}s  exact={em:.3f} cer={ce:.3f}")

elif phase == "grpo":
    CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    ST, CV, TOTAL = f"{TAG}_grpo_state.pt", f"{TAG}_grpo_curve.json", 1500
    base = f"{TAG}_sft_light.pt"
    m = T.Seq2Seq(len(Vs), len(Vt))
    ref = T.Seq2Seq(len(Vs), len(Vt)); ref.load_state_dict(torch.load(base))
    for p in ref.parameters(): p.requires_grad = False
    ref.eval()
    if os.path.exists(ST):
        s = torch.load(ST); m.load_state_dict(s["model"]); done = s["done"]
        curve = json.load(open(CV))
        torch.set_rng_state(s["rng"]); random.setstate(eval(s["pyrng"]))
    else:
        m.load_state_dict(torch.load(base)); done = 0
        em0, ce0 = T.evaluate(m, te, Vs, Vt)
        curve = [{"step": 0, "exact": em0, "cer": ce0, "reward": None}]
        print(f"[{TAG}] pre-GRPO exact={em0:.3f} cer={ce0:.3f}")
    run = min(CHUNK, TOTAL - done); t = time.time()
    m, c = T.grpo(m, tr, te[:30], Vs, Vt, steps=run, ref=ref, beta=0.04, log_every=50)
    for row in c: row["step"] += done; curve.append(row)
    done += run
    torch.save({"model": m.state_dict(), "done": done,
                "rng": torch.get_rng_state(), "pyrng": repr(random.getstate())}, ST)
    json.dump(curve, open(CV, "w"))
    em, ce = T.evaluate(m, te, Vs, Vt)
    print(f"[{TAG}] chunk {run} {time.time()-t:.0f}s done={done}/{TOTAL}  exact={em:.3f} cer={ce:.3f}")
    if done >= TOTAL:
        torch.save(m.state_dict(), f"{TAG}_grpo.pt")
        json.dump({"post": {"exact": em, "cer": ce}}, open(f"{TAG}_grpo_final.json", "w"))
        print(f"[{TAG}] GRPO COMPLETE")
