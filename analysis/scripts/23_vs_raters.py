"""次要分析：模型在本院 val 上与三位“评审员”（原记录 ASA、Claude 重标、Hermes 重标）的一致性，并与评审员两两一致性比较（模型作为另一位评审员）。
注意：Claude/Hermes 标签是指南锚定、独立于模型的重标，尚非 PI 仲裁后的参考标准；V 并入 IV 做四类。输出 reports/local_vs_raters.md"""
import os, glob, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
D = os.path.expanduser("~/research/NE_ERAS_JEV/data/derived"); RL = f"{D}/relabel"; R = os.path.expanduser("~/research/Anes_JEV_vs_LoRA/analysis/reports"); PC = ["p1", "p2", "p3", "p4"]
mp = pd.read_csv(f"{RL}/map.csv").rename(columns={"operation_id": "op_id"})
cl = pd.concat([pd.read_csv(f, sep="\t", header=None, names=["blind_id", "claude", "cc", "cr"]) for f in sorted(glob.glob(f"{RL}/labels/*.tsv"))])
hm = pd.read_csv(f"{RL}/agent/agent_labels.tsv", sep="\t").rename(columns={"盲ID": "blind_id", "agent_ASA": "hermes"})
c = pd.read_csv(f"{D}/asa_common_local.csv", usecols=["op_id", "split", "asa4"]).merge(mp[["blind_id", "op_id"]], on="op_id").merge(cl[["blind_id", "claude"]], on="blind_id").merge(hm[["blind_id", "hermes"]], on="blind_id")
for k in ("claude", "hermes"): c[k] = c[k].astype(str).str.extract(r"(\d)")[0].astype(int).clip(upper=4)
c = c[c.split == "val"].set_index("op_id")
def kap(a, b): return cohen_kappa_score(a, b, weights="quadratic")
out = [f"# 本院 val（n={len(c)}）：模型与三位评审员的一致性（二次加权 κ）\n"]
pairs = [("原记录", "claude", "asa4", "claude"), ("原记录", "hermes", "asa4", "hermes"), ("Claude", "hermes", "claude", "hermes")]
out += ["## 评审员两两\n", "| 配对 | κ |", "|---|---|"] + [f"| {a} – {b} | {kap(c[x], c[y]):.3f} |" for a, b, x, y in pairs]
def rd(f): p = f"{D}/{f}.csv"; return pd.read_csv(p).set_index("op_id") if os.path.exists(p) else None
M = {"读出零样本 T1": "pred_llm_local", "读出零样本 T2": "pred_llm_local_t2", "Clef-Flash 零样本": "pred_clef_local", "本院 LoRA T1 n=1000": "pred_lora_LOC1000s20261017_local_val", "本院 LoRA T1 n=4326": "pred_lora_LOC4326s20261017_local_val",
     "本院 LoRA T2 n=4326": "pred_lora_LOCT2_4326_local_val", "本院 Clef+LoRA T1 n=4326": "pred_clefLoRA_LCL4326_local_val", "MOVER-LoRA 12k 迁移": "pred_lora_mover12k_local_val"}
out += ["\n## 模型对各评审员的 κ\n", "| 模型 | 对原记录 | 对 Claude | 对 Hermes | 对三者均值 |", "|---|---|---|---|---|"]
for nm, f in M.items():
    d = rd(f)
    if d is None: continue
    ids = sorted(set(d.index) & set(c.index)); pr = d.loc[ids][PC].values.argmax(1) + 1; ks = [kap(c.loc[ids, r], pr) for r in ("asa4", "claude", "hermes")]
    out.append(f"| {nm} | {ks[0]:.3f} | {ks[1]:.3f} | {ks[2]:.3f} | {np.mean(ks):.3f} |")
open(f"{R}/local_vs_raters.md", "w").write("\n".join(out)); print("\n".join(out))
