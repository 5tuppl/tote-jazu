"""Gather per-phase metric files into a single results/results.json."""
import json, os, shutil
os.makedirs("results", exist_ok=True)

out = {
  "testbed": {"corpus_words": 420, "train": 336, "test": 84,
              "src_vocab": 29, "tgt_vocab": 40, "homograph_pairs": 5, "seed": 0},
  "held_out_full_test_n84": {
    "lookup_table":             json.load(open("m_lut.json"))["lut"],
    "sft_light_900":            json.load(open("m_sft_light.json"))["sft_light"],
    "sft_light_then_grpo_1500": json.load(open("m_grpo_final.json"))["post"],
    "sft_full_2500":            json.load(open("m_sft_full.json"))["sft_full"],
  },
  "grpo": {"steps": 1500, "group_size": 6, "kl_beta": 0.04,
           "temp_anneal": [1.1, 0.6], "reward": "(1 - CER) + 0.5*exact_match",
           "base_pre_grpo_full_test": {"exact": 0.036, "cer": 0.605},
           "monitoring_subset_n30_curve": "results/grpo_curve.json"},
  "note": "All numbers produced on CPU, fixed seed=0, reproducible via run_all.sh",
}
json.dump(out, open("results/results.json", "w"), indent=2)
if os.path.exists("grpo_curve.json"):
    shutil.copy("grpo_curve.json", "results/grpo_curve.json")
print("wrote results/results.json")
print(json.dumps(out["held_out_full_test_n84"], indent=2))
