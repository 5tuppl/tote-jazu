"""Build pairs.tsv: real Kazakh words (Cyrillic) -> Töte jazu, plus ambiguity stats.
Input: kk_words.txt (one Cyrillic word per line). Run download_words.sh first."""
import transliterate_tote as TT
from collections import Counter, defaultdict

words = open("kk_words.txt", encoding="utf-8").read().split()
ok = set(TT.CONS) | set(TT.VOWEL)
keep, seen = [], set()
for w in words:
    if 3 <= len(w) <= 10 and all(c in ok for c in w) and w not in seen:
        seen.add(w); keep.append(w)
keep = keep[:900]
pairs = [(TT.to_tote(w), w) for w in keep]
with open("pairs.tsv", "w", encoding="utf-8") as f:
    for ar, cy in pairs:
        f.write(f"{ar}\t{cy}\n")
print(f"wrote pairs.tsv ({len(pairs)} pairs)")

co = defaultdict(Counter)
for ar, cy in pairs:
    if len(ar) == len(cy):
        for a, c in zip(ar, cy): co[a][c] += 1
print("glyphs with >1 Cyrillic reading (the non-local ambiguity):")
for a, cnt in sorted(co.items(), key=lambda x: -sum(x[1].values())):
    if len(cnt) > 1:
        tot = sum(cnt.values())
        print(f"  {a!r}: {dict(cnt)}  majority={max(cnt.values())/tot:.0%}")
front = sum(1 for _, cy in pairs if TT.is_front_word(cy))
print(f"front-harmony words: {front}/{len(pairs)} ({front/len(pairs):.0%})")
