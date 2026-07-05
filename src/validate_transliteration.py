"""
validate_transliteration.py — externally validate our Töte jazu rules.

Our to_tote() rules already reproduce documented examples, but resting on one
implementation is weak. This script lets you cross-check against an INDEPENDENT
converter (e.g. nvrislam.net/k.php, alipbi.com). Most such converters are
client-side JavaScript, so there is usually no HTTP endpoint to call; the robust
path is a manual export:

  1. Run this script -> writes sample_for_review.csv (cyrillic, our_tote).
  2. Paste the cyrillic column into the online converter, copy its Töte output
     into a file reference_tote.tsv  (cyrillic <TAB> reference_tote).
  3. Re-run with that file -> prints exact-match agreement + per-char CER, and
     lists every disagreement so you (a native reader) can adjudicate.

Agreement near 100% validates our rules. Disagreements are exactly the cases a
reviewer would probe; resolving them strengthens the paper.
"""
import sys, csv, os
import transliterate_tote as TT

def cer(a, b):
    m, n = len(a), len(b); dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (a[i-1] != b[j-1])); prev = cur
    return dp[n] / max(1, len(b))

words = [w for w in open("kk_words.txt", encoding="utf-8").read().split()][:200]
ours = [(w, TT.to_tote(w)) for w in words]

ref_path = sys.argv[1] if len(sys.argv) > 1 else "reference_tote.tsv"
if not os.path.exists(ref_path):
    with open("sample_for_review.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f); wr.writerow(["cyrillic", "our_tote"])
        wr.writerows(ours)
    print("No reference file found.")
    print("Wrote sample_for_review.csv — paste the 'cyrillic' column into an")
    print("independent converter, save its output as reference_tote.tsv")
    print("(cyrillic<TAB>reference_tote), then re-run:")
    print(f"    python3 {sys.argv[0]} reference_tote.tsv")
    sys.exit(0)

ref = {}
for line in open(ref_path, encoding="utf-8"):
    if "\t" in line:
        c, t = line.rstrip("\n").split("\t"); ref[c] = t
ours_d = dict(ours)
common = [c for c in ours_d if c in ref]
exact = sum(ours_d[c] == ref[c] for c in common)
mean_cer = sum(cer(ours_d[c], ref[c]) for c in common) / max(1, len(common))
print(f"compared {len(common)} words")
print(f"exact agreement: {exact}/{len(common)} ({exact/len(common):.1%})")
print(f"mean per-char CER vs reference: {mean_cer:.3f}")
dis = [(c, ours_d[c], ref[c]) for c in common if ours_d[c] != ref[c]]
print(f"\n{len(dis)} disagreements (adjudicate these):")
for c, o, r in dis[:40]:
    print(f"  {c:12s} ours={o}   ref={r}")
