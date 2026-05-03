"""One-shot: extract loss trajectories from v9 JSON for clean display."""
import json
import numpy as np

data = json.load(open("analysis/v9_normalized_loss_data.json"))
ops = ["+", "-", "*", "/"]

print("=== v9 LOSS TRAJECTORY (best lr=1e-3, seed 0) — every 5000 steps ===")
print()
runs = data["part1_lr_sweep"]["1e-03"]
log = runs[0]["log"]
steps = log["step"]
losses = log["loss_per_op_raw"]
ks = log["k_per_op"]
print(f"  {'':>8}      | " + " ".join(f"loss_{op:<2s}".rjust(12) for op in ops) + "  ||  " + " ".join(f"k_{op:<2s}".rjust(8) for op in ops))
print(f"  {'step':>8}      | " + " ".join("rolling-200".rjust(12) for op in ops) + "  ||  " + " ".join("value".rjust(8) for op in ops))
print("  " + "-" * 120)
for i, s in enumerate(steps):
    if s % 5000 != 0 and s != steps[-1]:
        continue
    l = losses[i]
    k = ks[i]
    line = f"  {s:>8d}      | " + " ".join(f"{l[op]:>12.3f}" for op in ops) + "  ||  " + " ".join(f"{k[op]:>+8.4f}" for op in ops)
    print(line)

print()
print("=== AGGREGATE FINAL EVAL LOSS BY LR (5 seeds, raw MSE) ===")
print(f"   {'lr':>7}    {'+':>13}    {'-':>13}    {'*':>15}    {'/':>13}")
for lr_str in ["1e-04", "1e-03", "3e-03", "1e-02"]:
    runs = data["part1_lr_sweep"][lr_str]
    line = f"  {lr_str:>8}"
    for op in ops:
        vals = [r["eval_loss_per_op_raw"][op] for r in runs]
        m = np.mean(vals)
        s = np.std(vals)
        line += f"  {m:>9.2f}±{s:>5.2f}"
    print(line)

print()
print("=== AGGREGATE RATIO TO BASELINE BY LR ===")
print("   (lower=better; 0 = exact fit, 1 = as bad as predicting 0)")
print(f"   {'lr':>7}    {'+':>13}    {'-':>13}    {'*':>15}    {'/':>13}")
for lr_str in ["1e-04", "1e-03", "3e-03", "1e-02"]:
    runs = data["part1_lr_sweep"][lr_str]
    line = f"  {lr_str:>8}"
    for op in ops:
        vals = [r["eval_ratio_to_baseline"][op] for r in runs]
        m = np.mean(vals)
        s = np.std(vals)
        line += f"  {m:>7.4f}±{s:>6.4f}"
    print(line)

print()
print("=== BASELINES (predict-zero MSE per op) ===")
for op, b in data["baselines"].items():
    print(f"  {op}: {b:.4f}")
