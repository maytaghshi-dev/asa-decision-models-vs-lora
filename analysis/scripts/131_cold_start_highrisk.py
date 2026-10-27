"""Reviewer round (2026-10-28): (A) cold-start contrast at 100 (and 300) labels, calibrated decision model − LoRA,
label-matched (cutpoints fitted on exactly the LoRA training labels of each seed) and random-pool (label sampling + seed
variation in the uncertainty); (B) ASA IV–V operating characteristics (recall, precision, predicted positives, flags per
true positive) and 4×4 confusion matrices for κ-fitted cutpoints, accuracy-fitted cutpoints and LoRA@1,000.
Uses saved predictions only (no model is run). Output: analysis/reports_en/cold_start_highrisk.md"""
import os, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score as ck, confusion_matrix
D = os.path.expanduser("~/research/NE_ERAS_JEV/data/derived"); R = os.path.expanduser("~/research/Anes_JEV_vs_LoRA/analysis/reports_en")
PC = ["p1", "p2", "p3", "p4"]; SEEDS = ["20261017", "11", "12"]
K = lambda y, p: ck(y, p, weights="quadratic"); ACC = lambda y, p: (y == p).mean()
rd = lambda f: pd.read_csv(f"{D}/{f}.csv").set_index("op_id"); Ex = lambda a: (np.asarray(a) * np.arange(1, 5)).sum(1)
def fit(E, y, obj=K):
    cu = np.array([1.5, 2.5, 3.5]); g = np.linspace(1.0, 4.0, 121)
    for _ in range(3):
        for j in range(3): cu[j] = max(g, key=lambda v: obj(y, 1 + (E[:, None] > np.where(np.arange(3) == j, v, cu)).sum(1)))
    return cu
cut = lambda E, cu: 1 + (E[:, None] > cu).sum(1)
def clusters(pid, B, seed):
    up = np.unique(pid); idx = {p: np.where(pid == p)[0] for p in up}; rng = np.random.default_rng(seed)
    return [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(B)]
f3 = lambda v: "0.000" if abs(v) < 0.0005 else f"{v:+.3f}"
out = ["# Cold-start contrast and ASA IV–V operating characteristics (saved predictions only)\n"]

# ---------------- local data ----------------
c = pd.read_csv(f"{D}/asa_common_local.csv", usecols=["op_id", "split", "asa4"]).merge(
    pd.read_csv(f"{D}/asa_dataset_v1.csv", usecols=["operation_id", "patient_sn"]).rename(columns={"operation_id": "op_id"}), on="op_id").set_index("op_id")
full = pd.read_csv(f"{D}/asa_common_local.csv")                      # same frame and row order as 08_lora.local_df()
devdf = full[full.split == "dev"]; val = c[c.split == "val"].index
r = rd("pred_llm_local"); Ev = Ex(r.loc[val][PC].values); yv = c.loc[val].asa4.astype(int).values; pv = c.loc[val].patient_sn.values
Bc = clusters(pv, 1000, 20261028)
dev1k = c[c.split == "dev"].sample(1000, random_state=11).index        # main calibration pool

out += ["## A. Cold start: calibrated decision model − LoRA (local validation set, n = 2,753; T1)\n",
        "Label-matched: for each LoRA seed, the three cutpoints are fitted on exactly the n development labels that LoRA was trained on "
        "(`dev.sample(n, random_state=seed)`, as in 08_lora.py). Random-pool: 100 calibration draws of n labels from the 1,000-label pool (seed 11); "
        "each of 2,000 resamples draws patient clusters, one calibration draw and one LoRA seed. Intervals: patient-level cluster bootstrap.\n",
        "| n | Contrast | κ calibrated | κ LoRA | Difference (95% CI) |", "|---|---|---|---|---|"]
for n in (100, 300):
    lora = {s: rd(f"pred_lora_LOC{n}s{s}_local_val").loc[val][PC].values.argmax(1) + 1 for s in SEEDS}
    calm = {}
    for s in SEEDS:
        tr = devdf.sample(n, random_state=int(s)).op_id.values
        calm[s] = cut(Ev, fit(Ex(r.loc[tr][PC].values), c.loc[tr].asa4.astype(int).values))
    KB = lambda p: np.array([K(yv[b], p[b]) for b in Bc])
    kbC = {s: KB(calm[s]) for s in SEEDS}; kbL = {s: KB(lora[s]) for s in SEEDS}
    for s in SEEDS:
        d = kbC[s] - kbL[s]; out.append(f"| {n} | Label-matched, seed {s} | {K(yv, calm[s]):.3f} | {K(yv, lora[s]):.3f} | {f3(K(yv, calm[s]) - K(yv, lora[s]))} ({f3(np.percentile(d, 2.5))} to {f3(np.percentile(d, 97.5))}) |")
    d = np.mean([kbC[s] for s in SEEDS], 0) - np.mean([kbL[s] for s in SEEDS], 0)
    kc, kl = np.mean([K(yv, calm[s]) for s in SEEDS]), np.mean([K(yv, lora[s]) for s in SEEDS])
    out.append(f"| {n} | Label-matched, three-seed mean | {kc:.3f} | {kl:.3f} | {f3(kc - kl)} ({f3(np.percentile(d, 2.5))} to {f3(np.percentile(d, 97.5))}) |")
    rng = np.random.default_rng(100 + n); draws = []
    for _ in range(100):
        i = rng.choice(dev1k, n, replace=False); draws.append(cut(Ev, fit(Ex(r.loc[i][PC].values), c.loc[i].asa4.astype(int).values)))
    rng2 = np.random.default_rng(200 + n); Bc2 = clusters(pv, 2000, 300 + n); dd = []
    for b in Bc2:
        p = draws[rng2.integers(100)]; s = SEEDS[rng2.integers(3)]; dd.append(K(yv[b], p[b]) - K(yv[b], lora[s][b]))
    kc2 = np.mean([K(yv, p) for p in draws]); out.append(f"| {n} | Random pool (label sampling + seed variation) | {kc2:.3f} | {kl:.3f} | {f3(kc2 - kl)} ({f3(np.percentile(dd, 2.5))} to {f3(np.percentile(dd, 97.5))}) |")
    print(n, "done")

# ---------------- MOVER 100 labels (random pool only; LoRA training labels are not all in the scored pool) ----------------
sub = pd.read_csv(f"{D}/mover_test_sub4000.csv").op_id.tolist(); pm = pd.read_csv(f"{D}/mover_pid_map.csv").set_index("op_id").pid
rt = rd("pred_llm_mover_test"); rc = rd("pred_llm_mover_cal"); ids = [i for i in sub if i in rt.index]
Et = Ex(rt.loc[ids][PC].values); yt = rt.loc[ids].asa4.astype(int).values; pt = pm.loc[ids].values
Ec, yc = Ex(rc[PC].values), rc.asa4.astype(int).values
out += ["", "| n | Contrast (MOVER subset, n = 4,000) | κ calibrated | κ LoRA | Difference (95% CI) |", "|---|---|---|---|---|"]
for n in (100, 300):
    lora = [rd(f"pred_lora_curve{n}{sfx}_mover_sub").loc[ids][PC].values.argmax(1) + 1 for sfx in ("", "s11", "s12")]
    rng = np.random.default_rng(500 + n); draws = []
    for _ in range(100):
        i = rng.choice(len(yc), n, replace=False); draws.append(cut(Et, fit(Ec[i], yc[i])))
    rng2 = np.random.default_rng(600 + n); Bm = clusters(pt, 2000, 700 + n); dd = []
    for b in Bm:
        p = draws[rng2.integers(100)]; l = lora[rng2.integers(3)]; dd.append(K(yt[b], p[b]) - K(yt[b], l[b]))
    kc2 = np.mean([K(yt, p) for p in draws]); kl = np.mean([K(yt, l) for l in lora])
    out.append(f"| {n} | Random pool (label sampling + seed variation) | {kc2:.3f} | {kl:.3f} | {f3(kc2 - kl)} ({f3(np.percentile(dd, 2.5))} to {f3(np.percentile(dd, 97.5))}) |")
    print("mover", n, "done")

# ---------------- B. ASA IV–V operating characteristics ----------------
def op(y, p):
    tp = int(((p == 4) & (y == 4)).sum()); pp = int((p == 4).sum()); pos = int((y == 4).sum())
    return dict(recall=tp / pos, precision=(tp / pp if pp else float("nan")), pp=pp, tp=tp, fp=pp - tp, per1000=pp / len(y) * 1000, fpt=(pp / tp if tp else float("inf")), acc=ACC(y, p), k=K(y, p))
def row(site, arm, L):
    x = pd.DataFrame(L); m = x.mean(); lo, hi = x.min(), x.max()
    g = lambda k, d=2: f"{m[k]:.{d}f}" + (f" ({lo[k]:.{d}f}–{hi[k]:.{d}f})" if len(L) > 1 else "")
    return f"| {site} | {arm} | {g('k',3)} | {g('acc',3)} | {g('recall')} | {g('precision')} | {g('pp',0)} | {g('tp',0)} | {g('fp',0)} | {g('per1000',0)} | {g('fpt',1)} |"
out += ["", "## B. ASA IV–V operating characteristics\n",
        "Mean (min–max) over the 50 calibration pools of Supplementary S26 (same random generator) or over three LoRA seeds. "
        "Flags per true positive = predicted IV–V ÷ true IV–V among them (number needed to flag).\n",
        "| Data | Arm | κ | Accuracy | IV–V recall | IV–V precision | Predicted IV–V | True positives | False positives | Flagged per 1,000 | Flags per true positive |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
def pools(Ep, yp, Ee, ye, seed):
    rng = np.random.default_rng(seed); kk, aa = [], []
    for _ in range(50):
        i = rng.choice(len(yp), 1000, replace=False)
        kk.append(op(ye, cut(Ee, fit(Ep[i], yp[i], K)))); aa.append(op(ye, cut(Ee, fit(Ep[i], yp[i], ACC))))
    return kk, aa
dev = c[c.split == "dev"].index
kk, aa = pools(Ex(r.loc[dev][PC].values), c.loc[dev].asa4.astype(int).values, Ev, yv, 2026)
loraL = [rd(f"pred_lora_LOC1000s{s}_local_val").loc[val][PC].values.argmax(1) + 1 for s in SEEDS]
out += [row("Local val (IV–V n = %d)" % (yv == 4).sum(), "κ-fitted cutpoints, 50 pools", kk), row("Local val", "Accuracy-fitted cutpoints, 50 pools", aa), row("Local val", "LoRA@1,000, three seeds", [op(yv, p) for p in loraL])]
pool = rd("pred_pa_mover_calpool_orig")
kk, aa = pools(Ex(pool[PC].values), pool.asa4.astype(int).values, Et, yt, 2027)
loraM = [rd(f"pred_lora_curve1000{s}_mover_sub").loc[ids][PC].values.argmax(1) + 1 for s in ("", "s11", "s12")]
out += [row("MOVER (IV–V n = %d)" % (yt == 4).sum(), "κ-fitted cutpoints, 50 pools", kk), row("MOVER", "Accuracy-fitted cutpoints, 50 pools", aa), row("MOVER", "LoRA@1,000, three seeds", [op(yt, p) for p in loraM])]

# confusion matrices (main calibration pool, seed 11; LoRA summed over three seeds / 3)
def cm_md(title, y, p_list):
    m = np.mean([confusion_matrix(y, p, labels=[1, 2, 3, 4]) for p in p_list], 0)
    s = [f"\n**{title}** (rows: recorded class; columns: predicted class)\n", "| | I | II | III | IV–V |", "|---|---|---|---|---|"]
    for i, nm in enumerate(["I", "II", "III", "IV–V"]): s.append(f"| {nm} | " + " | ".join(f"{v:.0f}" if len(p_list) == 1 else f"{v:.1f}" for v in m[i]) + " |")
    return s
cuL = fit(Ex(r.loc[dev1k][PC].values), c.loc[dev1k].asa4.astype(int).values)
out += ["", "## C. Confusion matrices (local validation set)"]
out += cm_md("Calibrated decision model, κ-fitted cutpoints, main 1,000-label pool", yv, [cut(Ev, cuL)])
out += cm_md("LoRA@1,000, mean of three seeds", yv, loraL)
open(f"{R}/cold_start_highrisk.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
