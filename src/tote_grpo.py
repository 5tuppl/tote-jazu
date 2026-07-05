"""
tote_grpo.py — Verifiable-reward RL (GRPO) for post-OCR transliteration.

Task: Tote-jazu (Kazakh Arabic-script) character string -> Cyrillic Kazakh.
This is the post-OCR *transliteration* sub-task, deliberately small enough to
train on CPU. It exercises the exact RLVR machinery (GRPO + deterministic
verifier reward, no learned reward model) we care about.

The corpus here is a controlled SYNTHETIC cipher with intentional many-to-one
forward mappings, so the inverse (the task) is genuinely ambiguous and
context-dependent — mirroring real Tote-jazu (e.g. и/й and ы/і share Arabic
forms). On real aligned data (Quran parallel corpus) pass --data pairs.tsv.

Outputs: prints a results table and writes metrics_<tag>.json with the
held-out learning curve for plotting.
"""
import argparse, json, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2
SPECIAL = ["<pad>", "<bos>", "<eos>"]

# --------------------------------------------------------------------------- #
# Synthetic Cyrillic -> Arabic-script cipher (forward). Many-to-one on purpose.
# This is a controlled testbed, NOT verified Tote-jazu orthography.
# --------------------------------------------------------------------------- #
CYR = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюяіңғүұқөһ")
# A small Arabic-script letter pool; several Cyrillic letters collapse onto one
# Arabic glyph to inject genuine inverse-ambiguity.
AR_POOL = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
# Linguistically-motivated homographs: pairs that share one Arabic-script form
# in real Tote-jazu (vowel-harmony / semivowel collisions). Everything else is
# a clean bijection, so ambiguity is *local* and context-resolvable.
HOMOGRAPHS = [("и", "й"), ("ы", "і"), ("у", "ұ"), ("е", "э"), ("о", "ө")]
def build_cipher(seed=0):
    rng = random.Random(seed)
    pool = AR_POOL[:]
    rng.shuffle(pool)
    fwd = {}
    j = 0
    merged = {b: a for a, b in HOMOGRAPHS}     # second of pair -> first
    for c in CYR:
        if c in merged and merged[c] in fwd:
            fwd[c] = fwd[merged[c]]            # share the partner's glyph
        else:
            fwd[c] = pool[j % len(pool)]; j += 1
    return fwd
FWD = build_cipher()

def cyr_to_ar(s):  return "".join(FWD.get(c, c) for c in s)

# --------------------------------------------------------------------------- #
# Corpus: synthetic Kazakh-like Cyrillic words (deterministic generator).
# --------------------------------------------------------------------------- #
def make_corpus(n=420, seed=7):
    rng = random.Random(seed)
    cons = list("бвгджзйклмнпрстфхцчшқңғ")
    vow  = list("аеиоуыіүұөэяюё")
    words = set()
    while len(words) < n:
        L = rng.randint(3, 8)
        w = []
        for k in range(L):
            w.append(rng.choice(vow if k % 2 else cons))
        words.add("".join(w))
    words = sorted(words)
    pairs = [(cyr_to_ar(w), w) for w in words]   # (arabic, cyrillic gold)
    return pairs

# --------------------------------------------------------------------------- #
# Vocab
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = SPECIAL + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def enc(self, s): return [self.stoi[c] for c in s]
    def dec(self, ids):
        out = []
        for i in ids:
            if i == EOS: break
            if i in (PAD, BOS): continue
            out.append(self.itos[i])
        return "".join(out)

# --------------------------------------------------------------------------- #
# Model: GRU encoder + GRU decoder with dot-product attention
# --------------------------------------------------------------------------- #
class Seq2Seq(nn.Module):
    def __init__(self, src_v, tgt_v, emb=64, hid=128):
        super().__init__()
        self.src_emb = nn.Embedding(src_v, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_v, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.dec = nn.GRU(emb, hid, batch_first=True)
        self.bridge = nn.Linear(2*hid, hid)
        self.attn_proj = nn.Linear(2*hid, hid)
        self.out = nn.Linear(hid + hid, tgt_v)
        self.hid = hid
    def encode(self, src):
        mask = (src != PAD)
        e = self.src_emb(src)
        H, h = self.enc(e)                     # H:[B,T,2h]
        h = torch.tanh(self.bridge(torch.cat([h[0], h[1]], -1))).unsqueeze(0)
        return H, h, mask
    def step(self, tok, h, H, keys, mask):
        e = self.tgt_emb(tok)                  # [B,1,emb]
        o, h = self.dec(e, h)                  # o:[B,1,hid]
        scores = torch.bmm(keys, o.transpose(1, 2)).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(~mask, -1e9)
        a = torch.softmax(scores, -1).unsqueeze(1)               # [B,1,T]
        ctx = torch.bmm(a, H)                  # [B,1,2h]->but H is 2h; project
        return o, h, ctx
    def forward_logits(self, o, ctx_proj):
        return self.out(torch.cat([o.squeeze(1), ctx_proj.squeeze(1)], -1))

def run_decode(model, src, vocab_tgt, max_len=20, sample=False, temp=1.0):
    """Returns token id list (no specials) and summed logprob of chosen tokens."""
    H, h, mask = model.encode(src)
    keys = model.attn_proj(H)                  # [B,T,hid]
    B = src.size(0)
    tok = torch.full((B, 1), BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    outs = [[] for _ in range(B)]
    logp = torch.zeros(B)
    for _ in range(max_len):
        o, h, ctx = model.step(tok, h, H, keys, mask)
        ctx_proj = ctx[..., :model.hid] if ctx.size(-1) == model.hid else model.attn_proj(ctx)
        logits = model.forward_logits(o, ctx_proj)
        lp = F.log_softmax(logits / temp, -1)
        if sample:
            nxt = torch.multinomial(lp.exp(), 1).squeeze(1)
        else:
            nxt = lp.argmax(-1)
        logp = logp + lp.gather(1, nxt.unsqueeze(1)).squeeze(1) * (~done)
        for b in range(B):
            if not done[b]:
                if nxt[b].item() == EOS: done[b] = True
                else: outs[b].append(nxt[b].item())
        tok = nxt.unsqueeze(1)
        if done.all(): break
    return outs, logp

# --------------------------------------------------------------------------- #
# Metrics / reward
# --------------------------------------------------------------------------- #
def score_tokens(model, src, ids):
    """Summed log-prob of a given Cyrillic token id list under `model` (teacher
    forced). Differentiable if model params require grad; wrap in no_grad for ref."""
    H, h, mask = model.encode(src)
    keys = model.attn_proj(H)
    seq = [BOS] + ids + [EOS]
    total = src.new_zeros((), dtype=torch.float)
    for t in range(len(seq) - 1):
        tok = torch.tensor([[seq[t]]])
        o, h, ctx = model.step(tok, h, H, keys, mask)
        ctx_proj = model.attn_proj(ctx)
        logits = model.forward_logits(o, ctx_proj)
        total = total + F.log_softmax(logits, -1)[0, seq[t+1]]
    return total

def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (a[i-1] != b[j-1]))
            prev = cur
    return dp[n]

def cer(pred, gold):
    if not gold: return 0.0 if not pred else 1.0
    return edit_distance(pred, gold) / len(gold)

def reward(pred, gold):
    return (1.0 - cer(pred, gold)) + (0.5 if pred == gold else 0.0)

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, pairs, V_src, V_tgt):
    model.eval(); ex = 0; cers = []
    with torch.no_grad():
        for ar, cy in pairs:
            src = torch.tensor([V_src.enc(ar)], dtype=torch.long)
            ids, _ = run_decode(model, src, V_tgt, sample=False)
            pred = V_tgt.dec(ids[0])
            cers.append(cer(pred, cy))
            ex += (pred == cy)
    return ex / len(pairs), sum(cers) / len(cers)

# --------------------------------------------------------------------------- #
# SFT warm-start (teacher forcing)
# --------------------------------------------------------------------------- #
def sft(model, train, V_src, V_tgt, steps, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        ar, cy = random.choice(train)
        src = torch.tensor([V_src.enc(ar)], dtype=torch.long)
        tgt = [BOS] + V_tgt.enc(cy) + [EOS]
        H, h, mask = model.encode(src)
        keys = model.attn_proj(H)
        loss = 0.0
        for t in range(len(tgt) - 1):
            tok = torch.tensor([[tgt[t]]])
            o, h, ctx = model.step(tok, h, H, keys, mask)
            ctx_proj = ctx[..., :model.hid] if ctx.size(-1) == model.hid else model.attn_proj(ctx)
            logits = model.forward_logits(o, ctx_proj)
            loss = loss + F.cross_entropy(logits, torch.tensor([tgt[t+1]]))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# --------------------------------------------------------------------------- #
# GRPO: group-relative advantage, no value net, no learned reward
# --------------------------------------------------------------------------- #
def grpo(model, train, test, V_src, V_tgt, steps, G=6, lr=8e-4,
         t0=1.1, t1=0.6, lr_=None, ref=None, beta=0.04, log_every=40):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for s in range(steps):
        temp = t0 + (t1 - t0) * s / max(1, steps)
        ar, cy = random.choice(train)
        src = torch.tensor([V_src.enc(ar)], dtype=torch.long)
        logps, rews, kls = [], [], []
        model.train()
        for _ in range(G):
            ids, lp = run_decode(model, src, V_tgt, sample=True, temp=temp)
            pred = V_tgt.dec(ids[0])
            logps.append(lp); rews.append(reward(pred, cy))
            if ref is not None:
                lp_pol = score_tokens(model, src, ids[0])      # differentiable
                with torch.no_grad():
                    lp_ref = score_tokens(ref, src, ids[0])
                kls.append(lp_pol - lp_ref)                    # k1 KL estimator
        R = torch.tensor(rews)
        adv = (R - R.mean()) / (R.std() + 1e-6)                # group-relative
        lp = torch.stack(logps).squeeze(-1)
        loss = -(adv.detach() * lp).mean()
        if ref is not None and kls:
            loss = loss + beta * torch.stack(kls).mean()       # KL-to-reference
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            em, ce = evaluate(model, test, V_src, V_tgt)
            curve.append({"step": s, "exact": em, "cer": ce, "reward": R.mean().item()})
    return model, curve

def split(pairs, frac=0.8, seed=3):
    rng = random.Random(seed); idx = list(range(len(pairs))); rng.shuffle(idx)
    k = int(len(pairs) * frac)
    tr = [pairs[i] for i in idx[:k]]; te = [pairs[i] for i in idx[k:]]
    return tr, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--warmup", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    if a.data:
        pairs = []
        for line in open(a.data, encoding="utf-8"):
            if "\t" in line:
                x, y = line.rstrip("\n").split("\t"); pairs.append((x, y))
    else:
        pairs = make_corpus()
    V_src = Vocab([p[0] for p in pairs]); V_tgt = Vocab([p[1] for p in pairs])
    tr, te = split(pairs)
    print(f"corpus={len(pairs)} train={len(tr)} test={len(te)} "
          f"src_vocab={len(V_src)} tgt_vocab={len(V_tgt)}")

    def fresh():
        m = Seq2Seq(len(V_src), len(V_tgt)); 
        random.seed(a.seed); torch.manual_seed(a.seed); return m

    results = {}

    # 1) pure RL (no warm-start) — expected to cold-start fail
    torch.manual_seed(a.seed); random.seed(a.seed)
    m_rl = Seq2Seq(len(V_src), len(V_tgt))
    m_rl, c_rl = grpo(m_rl, tr, te, V_src, V_tgt, steps=a.steps)
    em, ce = evaluate(m_rl, te, V_src, V_tgt)
    results["pure_rl"] = {"exact": em, "cer": ce, "curve": c_rl}
    print(f"[pure GRPO]  held-out exact={em:.3f} CER={ce:.3f}")

    # 2) SFT only (baseline)
    torch.manual_seed(a.seed); random.seed(a.seed)
    m_sft = Seq2Seq(len(V_src), len(V_tgt))
    m_sft = sft(m_sft, tr, V_src, V_tgt, steps=a.warmup)
    em_s, ce_s = evaluate(m_sft, te, V_src, V_tgt)
    results["sft"] = {"exact": em_s, "cer": ce_s}
    print(f"[SFT only]   held-out exact={em_s:.3f} CER={ce_s:.3f}")

    # 3) SFT -> GRPO
    torch.manual_seed(a.seed); random.seed(a.seed)
    m = Seq2Seq(len(V_src), len(V_tgt))
    m = sft(m, tr, V_src, V_tgt, steps=a.warmup)
    em0, ce0 = evaluate(m, te, V_src, V_tgt)
    m, curve = grpo(m, tr, te, V_src, V_tgt, steps=a.steps)
    em1, ce1 = evaluate(m, te, V_src, V_tgt)
    results["sft_grpo"] = {"exact": em1, "cer": ce1, "exact0": em0, "cer0": ce0,
                           "curve": curve}
    print(f"[SFT->GRPO]  pre  exact={em0:.3f} CER={ce0:.3f}")
    print(f"[SFT->GRPO]  post exact={em1:.3f} CER={ce1:.3f}")

    json.dump(results, open(f"metrics_{a.tag}.json", "w"), indent=2)
    print(f"wrote metrics_{a.tag}.json")

if __name__ == "__main__":
    main()
