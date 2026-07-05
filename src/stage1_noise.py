"""Build noisy_pairs.tsv: simulate an OCR noise channel on the Töte jazu input.
Clean Cyrillic targets are unchanged (the OCR post-correction setting)."""
import random
random.seed(0)
RATE = 0.12   # per-character corruption probability
pairs = [ln.rstrip("\n").split("\t") for ln in open("pairs.tsv", encoding="utf-8") if "\t" in ln]
glyphs = sorted({c for ar, _ in pairs for c in ar})
def noise(s):
    out = []
    for ch in s:
        r = random.random()
        if r < RATE * 0.6:      out.append(random.choice(glyphs))  # substitution
        elif r < RATE * 0.8:    continue                            # deletion
        elif r < RATE:          out.append(ch); out.append(random.choice(glyphs))  # insertion
        else:                   out.append(ch)
    return "".join(out) or s
with open("noisy_pairs.tsv", "w", encoding="utf-8") as f:
    for ar, cy in pairs:
        f.write(f"{noise(ar)}\t{cy}\n")
print(f"wrote noisy_pairs.tsv ({len(pairs)} pairs, ~{RATE:.0%} per-char corruption)")
