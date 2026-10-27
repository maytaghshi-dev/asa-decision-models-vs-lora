"""fp16 LoRA (two-card layer split) at three seeds (local n = 1,000): kappa and accuracy per seed, difference from the 4-bit LoRA with the same labels, and H2 with the fp16 readout calibrated on pool 11
(label-matched only for seed 11; seeds 12 and 20261017 have partly overlapping pools). Patient-level cluster bootstrap B = 1000 (same as 112). Output reports_en/lora_fp16_xcard_3seed.md and reports/lora_fp16_3seed.json"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score as ck
D = os.path.expanduser("~/research/NE_ERAS_JEV/data/derived"); R = os.path.expanduser("~/research/Anes_JEV_vs_LoRA/analysis"); PC = ["p1", "p2", "p3", "p4"]
K = lambda y, p: ck(y, p, weights="quadratic"); rd = lambda f: pd.read_csv(f"{D}/{f}.csv").set_index("op_id"); Ex = lambda a: (np.asarray(a) * np.arange(1, 5)).sum(1)
def cuts_fit(E, y):
    cu = np.array([1.5, 2.5, 3.5]); g = np.linspace(1.0, 4.0, 121)
    for _ in range(3):
        for j in range(3): cu[j] = max(g, key=lambda v: K(y, 1 + (E[:, None] > np.where(np.arange(3) == j, v, cu)).sum(1)))
    return cu
c = pd.read_csv(f"{D}/asa_common_local.csv", usecols=["op_id", "split", "asa4"]).merge(pd.read_csv(f"{D}/asa_dataset_v1.csv", usecols=["operation_id", "patient_sn"]).rename(columns={"operation_id": "op_id"}), on="op_id").set_index("op_id")
val = c[c.split == "val"].index; y = c.loc[val].asa4.astype(int).values; pid = c.loc[val].patient_sn.values
cal = c[c.split == "dev"].reset_index().op_id.sample(1000, random_state=11).values; r = rd("pred_llm_local"); cu = cuts_fit(Ex(r.loc[cal][PC].values), c.loc[cal].asa4.astype(int).values)
cr = 1 + (Ex(r.loc[val][PC].values)[:, None] > cu).sum(1)
seeds = ["11", "12", "20261017"]; f16 = {s: rd(f"pred_lora_LOC1000s{s}fp16x2_local_val").loc[val][PC].values.argmax(1) + 1 for s in seeds}; q4 = {s: rd(f"pred_lora_LOC1000s{s}_local_val").loc[val][PC].values.argmax(1) + 1 for s in seeds}
up = np.unique(pid); ix = {q: np.where(pid == q)[0] for q in up}; rng = np.random.default_rng(8); B = [np.concatenate([ix[q] for q in rng.choice(up, len(up))]) for _ in range(1000)]
kb = lambda p: np.array([K(y[s], p[s]) for s in B]); KB = {("f", s): kb(f16[s]) for s in seeds}; KB.update({("q", s): kb(q4[s]) for s in seeds}); KB["cal"] = kb(cr)
ci = lambda v: f"{np.percentile(v, 2.5):+.3f} to {np.percentile(v, 97.5):+.3f}"
out = ["# fp16 LoRA across two cards at three seeds (local validation set, n = 2,753; local n = 1,000 labels)\n",
       "| Seed | fp16 LoRA κ (95% CI) | 4-bit LoRA κ (same labels) | fp16 − 4-bit (95% CI) | fp16 accuracy | Calibrated fp16 readout − fp16 LoRA (95% CI) | Label pool vs calibration pool |", "|---|---|---|---|---|---|---|"]
pool = {"11": "identical", "12": "partly overlapping (238/1000)", "20261017": "partly overlapping (223/1000)"}
for s in seeds:
    dd = KB["cal"] - KB[("f", s)]
    out.append(f"| {s} | {K(y, f16[s]):.3f} ({np.percentile(KB[('f', s)], 2.5):.3f}–{np.percentile(KB[('f', s)], 97.5):.3f}) | {K(y, q4[s]):.3f} | {K(y, f16[s]) - K(y, q4[s]):+.3f} ({ci(KB[('f', s)] - KB[('q', s)])}) | {(f16[s] == y).mean():.3f} | {K(y, cr) - K(y, f16[s]):+.3f} ({ci(dd)}) | {pool[s]} |")
mf = np.mean([KB[("f", s)] for s in seeds], 0); mq = np.mean([KB[("q", s)] for s in seeds], 0); kf = np.mean([K(y, f16[s]) for s in seeds]); kq = np.mean([K(y, q4[s]) for s in seeds]); d3 = KB["cal"] - mf
out += ["", f"- Three-seed mean of fp16 LoRA κ: {kf:.3f} (95% CI {np.percentile(mf, 2.5):.3f}–{np.percentile(mf, 97.5):.3f}); three-seed mean of 4-bit LoRA κ: {kq:.3f}; difference {kf - kq:+.3f} ({ci(mf - mq)}).",
        f"- **H2 with fp16 LoRA, three-seed mean** (fp16 readout calibrated on pool 11 minus mean of the three fp16 LoRA seeds): {K(y, cr) - kf:+.3f} ({ci(d3)}); lower bound > −0.05: {'yes' if np.percentile(d3, 2.5) > -0.05 else 'no'}.",
        f"- Reference, 4-bit LoRA three-seed mean (main analysis): {K(y, cr) - kq:+.3f} ({ci(KB['cal'] - mq)})."]
for s in seeds:
    J = f"{R}/reports/lora_fp16_xcard.json" if s == "11" else f"{R}/reports/lora_fp16_xcard_s{s}.json"; j = json.load(open(J)); out.append(f"- Seed {s}: training {j['train_s']:.0f} s, batch-1 latency p50 {j['p50_ms']:.0f} ms, peak training memory {j['peak_train_gb'][0]:.1f}/{j['peak_train_gb'][1]:.1f} GB.")
open(f"{R}/reports_en/lora_fp16_xcard_3seed.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
