"""Build the GRPO learning-curve figure (vector PDF) for the paper."""
import json, os, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

curve = json.load(open("grpo_curve.json"))
lut = json.load(open("m_lut.json"))["lut"]
sf  = json.load(open("m_sft_full.json"))["sft_full"]
steps = [r["step"] for r in curve]; cer = [r["cer"] for r in curve]; ex = [r["exact"] for r in curve]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(steps, cer, marker="o", ms=3, color="#1f4e79", label="SFT$\\to$GRPO")
ax[0].axhline(lut["cer"], ls="--", color="#888", label="Lookup table")
ax[0].axhline(sf["cer"],  ls=":",  color="#c0504d", label="SFT (full)")
ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("CER"); ax[0].set_title("(a) Character error rate")
ax[0].legend(fontsize=6, loc="upper right"); ax[0].grid(alpha=.3)
ax[1].plot(steps, ex, marker="o", ms=3, color="#1f4e79")
ax[1].axhline(lut["exact"], ls="--", color="#888")
ax[1].axhline(sf["exact"],  ls=":",  color="#c0504d")
ax[1].set_xlabel("GRPO step"); ax[1].set_ylabel("Exact-match"); ax[1].set_title("(b) Exact-match accuracy")
ax[1].grid(alpha=.3)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("paper/curve.pdf", bbox_inches="tight")
shutil.copy("paper/curve.pdf", "results/curve.pdf")
print("wrote paper/curve.pdf and results/curve.pdf")
