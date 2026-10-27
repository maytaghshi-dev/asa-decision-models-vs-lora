"""Calibration-label curve for the chat-template readout (local validation set): k labels drawn from the seed-11 1,000-label pool, 100 repetitions, k = 0, 25, 50, 100, 200, 500, 1000.
Compares with the raw-prompt readout curve (same procedure, 121-point grid, 3 passes). Output reports_en/chat_readout_curve.md"""
import os, numpy as np, pandas as pd, torch, sys
sys.argv = [sys.argv[0], "0"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "126_chat_readout.py")).read().split("c = pd.read_csv(f\"{D}/asa_common_local.csv\")")[0])
c = pd.read_csv(f"{D}/asa_common_local.csv"); val = c[c.split == "val"].reset_index(drop=True); cal = c[c.split == "dev"].sample(1000, random_state=11).reset_index(drop=True)
pc = probs(list(cal.text)); pv = pd.read_csv(f"{D}/pred_llm_local_chat.csv").set_index("op_id").loc[val.op_id][PC].values
old = pd.read_csv(f"{D}/pred_llm_local.csv").set_index("op_id"); oc = old.loc[cal.op_id][PC].values; ov = old.loc[val.op_id][PC].values
y = val.asa4.astype(int).values; yc = cal.asa4.astype(int).values; rng = np.random.default_rng(7); draws = [rng.choice(1000, k, replace=False) for k in (25, 50, 100, 200, 500) for _ in range(100)]
def curve(Ec, Ev):
    res = {0: K(y, 1 + (Ev[:, None] > np.array([1.5, 2.5, 3.5])).sum(1))}; i = 0
    for k in (25, 50, 100, 200, 500):
        v = []
        for _ in range(100): ix = draws[i]; i += 1; v.append(K(y, 1 + (Ev[:, None] > fit(Ec[ix], yc[ix])).sum(1)))
        res[k] = v
    res[1000] = [K(y, 1 + (Ev[:, None] > fit(Ec, yc)).sum(1))]; return res
f = lambda v: f"{np.mean(v):.3f} ({np.percentile(v, 2.5):.3f}–{np.percentile(v, 97.5):.3f})" if len(v) > 1 else f"{v[0]:.3f}"
a = curve(Ex(oc), Ex(ov)); b = curve(Ex(pc), Ex(pv)); a[0] = [a[0]]; b[0] = [b[0]]
rows = ["# Calibration-label curve: raw-prompt readout versus chat-template readout (local validation set, n = 2,753)\n", "Mean (2.5–97.5 percentile) of κ over 100 draws of k labels from the seed-11 pool; k = 0 is the uncalibrated argmax; k = 1,000 is the whole pool. The raw-prompt values use the same fine grid as the main analysis (0.534 at k = 1,000).\n",
        "| k | Raw-prompt readout (main analysis) | Chat-template readout |", "|---|---|---|"]
for k in (0, 25, 50, 100, 200, 500, 1000): rows.append(f"| {k} | {f(a[k])} | {f(b[k])} |")
open(f"{R}/chat_readout_curve.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
