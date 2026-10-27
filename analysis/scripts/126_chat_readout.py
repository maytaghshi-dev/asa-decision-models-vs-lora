"""Readout under the chat template (thinking off) with the instruction "Reply with the option letter only.": softmax over the logits of the four option-letter tokens at the first generated position.
Local validation set (n = 2,753) and the 1,000-label calibration pool (seed 11). Reports uncalibrated (argmax) and calibrated kappa/accuracy, and the agreement with the answer-only chat8 generation on the 1,000-case H1 subset.
Output reports_en/chat_readout.md; derived/pred_llm_local_chat.csv"""
import os, sys, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score as ck
gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1; D = os.path.expanduser("~/research/NE_ERAS_JEV/data/derived"); R = os.path.expanduser("~/research/Anes_JEV_vs_LoRA/analysis/reports_en"); PC = ["p1", "p2", "p3", "p4"]
P = "You are an anesthesiologist. Based on the preoperative information below, assign the ASA Physical Status class.\n\n%s\n\nOptions:\nA. ASA I\nB. ASA II\nC. ASA III\nD. ASA IV-V\nAnswer:"
SUFFIX = "\n\nReply with the option letter only."
path = "/mnt/c/backup/Qwen3.5-9B"; tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map={"": gpu}).eval(); dev = f"cuda:{gpu}"; ids = torch.tensor([tok.encode(c, add_special_tokens=False)[-1] for c in "ABCD"])
def build(t): return tok.apply_chat_template([{"role": "user", "content": (P % t).rsplit("Answer:", 1)[0].rstrip() + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def probs(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        x = tok([build(t) for t in texts[i:i + bs]], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad(): lg = m(**x, logits_to_keep=1).logits[:, -1].float().cpu()
        out.append(torch.softmax(lg[:, ids], -1).numpy())
    return np.concatenate(out)
K = lambda y, p: ck(y, p, weights="quadratic"); Ex = lambda a: (np.asarray(a) * np.arange(1, 5)).sum(1)
def fit(E, y):
    cu = np.array([1.5, 2.5, 3.5]); g = np.linspace(1.0, 4.0, 121)
    for _ in range(3):
        for j in range(3): cu[j] = max(g, key=lambda v: K(y, 1 + (E[:, None] > np.where(np.arange(3) == j, v, cu)).sum(1)))
    return cu
c = pd.read_csv(f"{D}/asa_common_local.csv"); val = c[c.split == "val"].reset_index(drop=True); cal = c[c.split == "dev"].sample(1000, random_state=11).reset_index(drop=True)
pv = probs(list(val.text)); pc = probs(list(cal.text)); pd.DataFrame(dict(op_id=val.op_id, asa4=val.asa4, p1=pv[:, 0], p2=pv[:, 1], p3=pv[:, 2], p4=pv[:, 3])).to_csv(f"{D}/pred_llm_local_chat.csv", index=False)
y = val.asa4.astype(int).values; cu = fit(Ex(pc), cal.asa4.astype(int).values); pa = pv.argmax(1) + 1; pcal = 1 + (Ex(pv)[:, None] > cu).sum(1)
old = pd.read_csv(f"{D}/pred_llm_local.csv").set_index("op_id").loc[val.op_id][PC].values; oa = old.argmax(1) + 1; ocal = 1 + (Ex(old)[:, None] > fit(Ex(pd.read_csv(f"{D}/pred_llm_local.csv").set_index("op_id").loc[cal.op_id][PC].values), cal.asa4.astype(int).values)).sum(1)
sub = c[c.split == "val"].sample(1000, random_state=20261019).op_id.tolist(); idx = val.op_id.isin(sub).values; g = pd.read_csv(f"{D}/pred_ans_local_chat8.csv").set_index("op_id").loc[val.op_id[idx]].pred.values
rows = ["# Readout under the chat template (letter-token probabilities at the first generated position; local validation set, n = 2,753)\n",
 "| Input format | Uncalibrated κ | Uncalibrated accuracy | κ after 1,000-label calibration | Accuracy after calibration | Macro-class predicted share, uncalibrated I / II / III / IV–V (%) |", "|---|---|---|---|---|---|"]
sh = lambda p: " / ".join(f"{(p == k).mean()*100:.1f}" for k in range(1, 5))
rows.append(f"| Raw readout prompt (main analysis) | {K(y, oa):.3f} | {(oa == y).mean():.3f} | {K(y, ocal):.3f} | {(ocal == y).mean():.3f} | {sh(oa)} |")
rows.append(f"| Chat template + \"Reply with the option letter only.\" | {K(y, pa):.3f} | {(pa == y).mean():.3f} | {K(y, pcal):.3f} | {(pcal == y).mean():.3f} | {sh(pa)} |")
rows += ["", f"- Agreement between the chat-template readout argmax and the chat8 answer-only generation on the 1,000-case H1 subset: {(pa[idx] == g).mean():.3f} (n = {idx.sum()}).",
         f"- κ on the H1 subset: chat-template readout argmax {K(y[idx], pa[idx]):.3f}; chat8 generation {K(y[idx], g):.3f}; raw readout argmax {K(y[idx], oa[idx]):.3f}."]
open(f"{R}/chat_readout.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
