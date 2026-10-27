"""Per-class recall and predicted-class share of zero-shot readout, kappa-fitted and accuracy-fitted cutpoint calibration, and LoRA@1000 (local validation set and MOVER subset).
Calibration labels and cutpoint search as in 99_adjacent.py (pool seed 11, grid 121, 3 passes); the accuracy-fitted variant maximizes exact accuracy instead of quadratic-weighted kappa.
Output: reports_en/class_recall.md"""
import os, numpy as np, pandas as pd
D = os.path.expanduser("~/research/NE_ERAS_JEV/data/derived"); R = os.path.expanduser("~/research/Anes_JEV_vs_LoRA/analysis/reports_en"); PC = ["p1", "p2", "p3", "p4"]
from sklearn.metrics import cohen_kappa_score as ck
K = lambda y, p: ck(y, p, weights="quadratic"); ACC = lambda y, p: (y == p).mean(); rd = lambda f: pd.read_csv(f"{D}/{f}.csv").set_index("op_id"); Ex = lambda a: (np.asarray(a) * np.arange(1, 5)).sum(1)
def fit(E, y, obj):
    cu = np.array([1.5, 2.5, 3.5]); g = np.linspace(1.0, 4.0, 121)
    for _ in range(3):
        for j in range(3): cu[j] = max(g, key=lambda v: obj(y, 1 + (E[:, None] > np.where(np.arange(3) == j, v, cu)).sum(1)))
    return cu
cut = lambda E, cu: 1 + (E[:, None] > cu).sum(1)
def rec(y, p): return [((p == k) & (y == k)).sum() / max((y == k).sum(), 1) for k in range(1, 5)]
def share(p): return [(p == k).mean() * 100 for k in range(1, 5)]
def row(site, nm, y, ps):
    ps = ps if isinstance(ps, list) else [ps]
    r = np.mean([rec(y, p) for p in ps], 0); s = np.mean([share(p) for p in ps], 0); k = np.mean([K(y, p) for p in ps]); a = np.mean([ACC(y, p) for p in ps])
    return f"| {site} | {nm} | {k:.3f} | {a:.3f} | " + " / ".join(f"{x:.2f}" for x in r) + " | " + " / ".join(f"{x:.1f}" for x in s) + " |"
out = ["# Per-class recall and predicted-class share (ASA I / II / III / IV–V)\n",
       "Calibration pool: 1,000 development labels (seed 11); cutpoint search: initial cutpoints 1.5/2.5/3.5, grid of 121 points over [1, 4], 3 coordinate passes. κ-fitted cutpoints maximize quadratic-weighted κ (main analysis); accuracy-fitted cutpoints maximize exact accuracy. LoRA is the mean over three seeds at n = 1,000 (argmax). Recall is given per class as I / II / III / IV–V; share is the percentage of predictions per class.\n",
       "| Data | Arm | κ | Accuracy | Recall I / II / III / IV–V | Predicted share I / II / III / IV–V (%) |", "|---|---|---|---|---|---|"]
c = pd.read_csv(f"{D}/asa_common_local.csv", usecols=["op_id", "split", "asa4"]).set_index("op_id"); val = c[c.split == "val"].index; dev = c[c.split == "dev"].sample(1000, random_state=11).index
y = c.loc[val].asa4.astype(int).values; yd = c.loc[dev].asa4.astype(int).values; r = rd("pred_llm_local"); Ev = Ex(r.loc[val][PC].values); Ed = Ex(r.loc[dev][PC].values)
out.append(f"| Local val | True distribution | — | — | — | " + " / ".join(f"{x:.1f}" for x in share(y)) + " |")
out.append(row("Local val", "Readout zero-shot (argmax)", y, r.loc[val][PC].values.argmax(1) + 1))
out.append(row("Local val", "Readout + 1,000 labels, κ-fitted cutpoints", y, cut(Ev, fit(Ed, yd, K))))
out.append(row("Local val", "Readout + 1,000 labels, accuracy-fitted cutpoints", y, cut(Ev, fit(Ed, yd, ACC))))
out.append(row("Local val", "LoRA n = 1,000 (three-seed mean)", y, [rd(f"pred_lora_LOC1000s{s}_local_val").loc[val][PC].values.argmax(1) + 1 for s in ("20261017", "11", "12")]))
sub = pd.read_csv(f"{D}/mover_test_sub4000.csv").op_id.tolist(); rt = rd("pred_llm_mover_test"); rc = rd("pred_llm_mover_cal"); ids = [i for i in sub if i in rt.index]
y = rt.loc[ids].asa4.astype(int).values; yc = rc.asa4.astype(int).values; Et = Ex(rt.loc[ids][PC].values); Ec = Ex(rc[PC].values)
out.append(f"| MOVER subset | True distribution | — | — | — | " + " / ".join(f"{x:.1f}" for x in share(y)) + " |")
out.append(row("MOVER subset", "Readout zero-shot (argmax)", y, rt.loc[ids][PC].values.argmax(1) + 1))
out.append(row("MOVER subset", "Readout + 1,000 labels, κ-fitted cutpoints", y, cut(Et, fit(Ec, yc, K))))
out.append(row("MOVER subset", "Readout + 1,000 labels, accuracy-fitted cutpoints", y, cut(Et, fit(Ec, yc, ACC))))
out.append(row("MOVER subset", "LoRA n = 1,000 (three-seed mean)", y, [rd(f"pred_lora_curve1000{s}_mover_sub").loc[ids][PC].values.argmax(1) + 1 for s in ("", "s11", "s12")]))
open(f"{R}/class_recall.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
