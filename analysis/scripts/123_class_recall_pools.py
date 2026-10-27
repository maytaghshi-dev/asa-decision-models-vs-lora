"""S26 extension: 50 independent 1,000-label pools -> kappa-fitted and accuracy-fitted cutpoints -> accuracy, kappa, class I and IV-V recall on the evaluation set.
Local: pools from the 4,326 development operations; MOVER: pools from the further 5,000 training-split operations (pred_pa_mover_calpool_orig). Output reports_en/class_recall_pools.md"""
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
rec = lambda y, p, k: ((p == k) & (y == k)).sum() / max((y == k).sum(), 1)
def stats(y, p): return [K(y, p), ACC(y, p), rec(y, p, 1), rec(y, p, 4), (p == 4).mean() * 100]
def run(site, Ep, yp, Ev, yv, lora, seed):
    rng = np.random.default_rng(seed); res = {"k": [], "a": []}
    for _ in range(50):
        i = rng.choice(len(yp), 1000, replace=False)
        res["k"].append(stats(yv, cut(Ev, fit(Ep[i], yp[i], K)))); res["a"].append(stats(yv, cut(Ev, fit(Ep[i], yp[i], ACC))))
    out = []
    for nm, key in (("κ-fitted cutpoints", "k"), ("accuracy-fitted cutpoints", "a")):
        x = np.array(res[key]); f = lambda j, d=3: f"{x[:, j].mean():.{d}f} ({x[:, j].min():.{d}f}–{x[:, j].max():.{d}f})"
        out.append(f"| {site} | Readout, 50 pools, {nm} | {f(0)} | {f(1)} | {f(2, 2)} | {f(3, 2)} | {f(4, 1)} |")
    l = np.array([stats(yv, p) for p in lora]); g = lambda j, d=3: f"{l[:, j].mean():.{d}f} ({l[:, j].min():.{d}f}–{l[:, j].max():.{d}f})"
    out.append(f"| {site} | LoRA n = 1,000, three seeds | {g(0)} | {g(1)} | {g(2, 2)} | {g(3, 2)} | {g(4, 1)} |"); return out
rows = ["# Per-class recall over 50 independent 1,000-label calibration pools\n",
        "Mean (min–max) over the same 50 pools as in Supplementary S18, drawn without replacement from the development operations (local: 4,326; MOVER: a further 5,000 training-split operations); the pools overlap, so the range understates between-pool variation. LoRA: mean (min–max) over three seeds. Evaluation sets: local validation set (n = 2,753; ASA I n = 36, IV–V n = 80) and MOVER subset (n = 4,000).\n",
        "| Data | Arm | κ | Accuracy | Recall class I | Recall class IV–V | Predicted share IV–V (%) |", "|---|---|---|---|---|---|---|"]
c = pd.read_csv(f"{D}/asa_common_local.csv", usecols=["op_id", "split", "asa4"]).set_index("op_id"); val = c[c.split == "val"].index; dev = c[c.split == "dev"].index
r = rd("pred_llm_local"); lora = [rd(f"pred_lora_LOC1000s{s}_local_val").loc[val][PC].values.argmax(1) + 1 for s in ("20261017", "11", "12")]
rows += run("Local val", Ex(r.loc[dev][PC].values), c.loc[dev].asa4.astype(int).values, Ex(r.loc[val][PC].values), c.loc[val].asa4.astype(int).values, lora, 2026)
sub = pd.read_csv(f"{D}/mover_test_sub4000.csv").op_id.tolist(); rt = rd("pred_llm_mover_test"); ids = [i for i in sub if i in rt.index]; pool = rd("pred_pa_mover_calpool_orig")
lora = [rd(f"pred_lora_curve1000{s}_mover_sub").loc[ids][PC].values.argmax(1) + 1 for s in ("", "s11", "s12")]
rows += run("MOVER subset", Ex(pool[PC].values), pool.asa4.astype(int).values, Ex(rt.loc[ids][PC].values), rt.loc[ids].asa4.astype(int).values, lora, 2027)
open(f"{R}/class_recall_pools.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
