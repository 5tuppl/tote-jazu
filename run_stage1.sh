#!/usr/bin/env bash
# Reproduce Stage 1 (real Kazakh data) end-to-end on CPU. ~15 min total.
set -e
cd "$(dirname "$0")"
echo "[1/6] download real Kazakh words"; bash download_words.sh
echo "[2/6] validate transliterator";    python3 src/transliterate_tote.py
echo "[3/6] build clean + noisy corpora"; python3 src/stage1_build.py; python3 src/stage1_noise.py
echo "[4/6] CLEAN: lookup + SFT + GRPO"
python3 src/stage1_pipeline.py lut
python3 src/stage1_pipeline.py sft_light
python3 src/stage1_pipeline.py sft_full
rm -f s1_grpo_state.pt s1_grpo_curve.json
for i in 1 2 3 4 5; do python3 src/stage1_pipeline.py grpo 300; done
echo "[5/6] NOISY: lookup + SFT + GRPO"
export PAIRS=noisy_pairs.tsv TAG=s1n
python3 src/stage1_pipeline.py lut
python3 src/stage1_pipeline.py sft_light
python3 src/stage1_pipeline.py sft_full
rm -f s1n_grpo_state.pt s1n_grpo_curve.json
for i in 1 2 3 4 5; do python3 src/stage1_pipeline.py grpo 300; done
unset PAIRS TAG
echo "[6/6] done. See stage1_results.json"
