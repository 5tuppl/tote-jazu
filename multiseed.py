"""
multiseed.py — self-driving, resumable 5-seed variance study for Stage 1.
Place in tote-jazu-rlvr/src/ (next to tote_grpo.py) and run:  python3 multiseed.py
Re-run until it prints ALL DONE. Progress is saved after every unit.

Settings (consistent across all methods/seeds): SFT_LIGHT=900, SFT_FULL=2500,
GRPO=600 (2x300 chunks, KL beta=0.04, group size G=6).
"""
import os, sys, json, time, random, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # so `import tote_grpo` works
import torch
import tote_grpo as T

def find(name):
    """Locate a data file anywhere in the bundle."""
    cands = [os.path.join(HERE, name), os.path.join(HERE, "..", name),
             os.path.join(HERE, "..", "data", name), os.path.join("data", name), name]
    for c in cands:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(HERE, "..", "**", name), recursive=True)
    if hits:
        return hits[0]
    sys.exit(f"Could not find {name}. Make sure the bundle's data/ folder is present.")

CONDS = {"clean": find("pairs.tsv"), "noisy": find("noisy_pairs.tsv")}
SEEDS = [0, 1, 2, 3, 4]
SFT_LIGHT, SFT_FULL, GRPO_STEPS, GRPO_CHUNK = 900, 2500, 600, 300
BUDGET = 120                                  # seconds of work per invocation

RUNDIR = os.path.join(HERE, "ms_runs")
os.makedirs(RUNDIR, exist_ok=True)
STATE = os.path.join(RUNDIR, "multiseed_state.json")
ck = lambda *p: os.path.join(RUNDIR, "_".join(map(str, p)))

def load_pairs(path):
    return [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8") if "\t" in l]

def prep(cond, seed):
    pairs = load_pairs(CONDS[cond])
    Vs = T.Vocab([p[0] for p in pairs]); Vt = T.Vocab([p[1] for p in pairs])
    tr, te = T.split(pairs, seed=seed)        # vary split with seed
    return pairs, Vs, Vt, tr, te

def st_load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"done": {}, "results": {}}

def st_save(s):
    json.dump(s, open(STATE, "w"), indent=1)

def key(seed, cond, m):
    return f"{seed}_{cond}_{m}"

UNITS = []
for cond in CONDS:
    for seed in SEEDS:
        UNITS += [(seed, cond, "lut"), (seed, cond, "sft_light"),
                  (seed, cond, "sft_full"), (seed, cond, "grpo")]

def do_lut(seed, cond):
    from collections import Counter, defaultdict
    _, _, _, tr, te = prep(cond, seed)
    co = defaultdict(Counter)
    for ar, cy in tr:
        if len(ar) == len(cy):
            for a, c in zip(ar, cy): co[a][c] += 1
    inv = {a: cnt.most_common(1)[0][0] for a, cnt in co.items()}
    f = lambda s: "".join(inv.get(a, a) for a in s)
    ex = sum(f(ar) == cy for ar, cy in te)
    ce = sum(T.cer(f(ar), cy) for ar, cy in te) / len(te)
    return {"exact": ex / len(te), "cer": ce}

def do_sft(seed, cond, steps, tag):
    pairs, Vs, Vt, tr, te = prep(cond, seed)
    random.seed(seed); torch.manual_seed(seed)
    m = T.Seq2Seq(len(Vs), len(Vt))
    m = T.sft(m, tr, Vs, Vt, steps=steps, lr=1e-3)
    em, ce = T.evaluate(m, te, Vs, Vt)
    torch.save(m.state_dict(), ck(seed, cond, tag) + ".pt")
    return {"exact": em, "cer": ce}

def do_grpo_chunk(seed, cond):
    pairs, Vs, Vt, tr, te = prep(cond, seed)
    base = ck(seed, cond, "sft_light") + ".pt"
    stf = ck(seed, cond, "grpo_state") + ".pt"
    m = T.Seq2Seq(len(Vs), len(Vt))
    ref = T.Seq2Seq(len(Vs), len(Vt)); ref.load_state_dict(torch.load(base))
    for p in ref.parameters(): p.requires_grad = False
    ref.eval()
    if os.path.exists(stf):
        s = torch.load(stf); m.load_state_dict(s["model"]); done = s["done"]
        torch.set_rng_state(s["rng"]); random.setstate(eval(s["pyrng"]))
    else:
        m.load_state_dict(torch.load(base)); done = 0
        random.seed(seed); torch.manual_seed(seed)
    run = min(GRPO_CHUNK, GRPO_STEPS - done)
    m, _ = T.grpo(m, tr, te[:30], Vs, Vt, steps=run, ref=ref, beta=0.04, log_every=999)
    done += run
    torch.save({"model": m.state_dict(), "done": done,
                "rng": torch.get_rng_state(), "pyrng": repr(random.getstate())}, stf)
    if done >= GRPO_STEPS:
        em, ce = T.evaluate(m, te, Vs, Vt)
        return True, {"exact": em, "cer": ce}
    return False, None

def summarize(s):
    import statistics as st
    methods = ["lut", "sft_light", "sft_full", "grpo"]
    name = {"lut": "Lookup table", "sft_light": "SFT (light)",
            "sft_full": "SFT (full)", "grpo": "SFT->GRPO"}
    print("\n=== 5-seed mean +/- std (CER / exact-match) ===")
    for cond in CONDS:
        print(f"\n[{cond}]")
        for m in methods:
            vals = [s["results"][key(sd, cond, m)] for sd in SEEDS
                    if key(sd, cond, m) in s["results"]]
            if len(vals) < 2:
                continue
            cer = [v["cer"] for v in vals]; ex = [v["exact"] for v in vals]
            print(f"  {name[m]:14s} CER {st.mean(cer):.3f}+/-{st.pstdev(cer):.3f}"
                  f"   exact {st.mean(ex):.3f}+/-{st.pstdev(ex):.3f}  (n={len(vals)})")

def main():
    s = st_load(); t0 = time.time()
    for seed, cond, m in UNITS:
        unit = key(seed, cond, m)
        if s["done"].get(unit):
            continue
        if time.time() - t0 > BUDGET:
            st_save(s)
            print(f"...budget reached, {sum(s['done'].values())}/{len(UNITS)} units done. re-run.")
            return
        if m == "lut":
            s["results"][unit] = do_lut(seed, cond); s["done"][unit] = 1
        elif m == "sft_light":
            s["results"][unit] = do_sft(seed, cond, SFT_LIGHT, "sft_light"); s["done"][unit] = 1
        elif m == "sft_full":
            s["results"][unit] = do_sft(seed, cond, SFT_FULL, "sft_full"); s["done"][unit] = 1
        elif m == "grpo":
            fin, met = do_grpo_chunk(seed, cond)
            if fin:
                s["results"][unit] = met; s["done"][unit] = 1
        st_save(s)
        print(f"  {unit}: {s['results'].get(unit, 'grpo chunk done, more to go')}")
    if len(s["done"]) >= len(UNITS):
        print("ALL DONE")
        summarize(s)
    else:
        print(f"progress {sum(s['done'].values())}/{len(UNITS)} units; re-run.")

if __name__ == "__main__":
    main()
