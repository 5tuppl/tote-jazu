#!/usr/bin/env bash
# Reproduce every number in results.json from scratch on CPU. seed=0 throughout.
# Total runtime ~8-10 min on a laptop CPU. No GPU needed.
set -e
cd "$(dirname "$0")"

echo "[1/5] supervised baselines (light + full)"
python3 src/driver.py sft_light          # -> sft_light.pt, m_sft_light.json
python3 src/driver.py sft_full           # -> sft_full.pt,  m_sft_full.json

echo "[2/5] deterministic lookup-table baseline"
python3 src/lut_baseline.py              # -> m_lut.json

echo "[3/5] GRPO / RLVR from the light SFT base (chunked, resumable)"
rm -f grpo_state.pt grpo_curve.json
for i in 1 2 3 4 5; do
  python3 src/grpo_chunk.py 300          # 5 x 300 = 1500 steps -> grpo.pt, m_grpo_final.json
done

echo "[4/5] consolidate results.json"
python3 src/consolidate.py

echo "[5/5] figure + paper"
python3 src/make_figure.py               # -> results/curve.pdf
( cd paper && pdflatex -interaction=nonstopmode paper.tex >/dev/null \
            && pdflatex -interaction=nonstopmode paper.tex >/dev/null )
echo "DONE. See results/results.json and paper/paper.pdf"
