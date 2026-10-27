"""量化对照（用户提问）：(a) 4-bit 量化的零样本读出在本院 val + 校准池上的 κ；(b) 同一 LoRA 适配器（本院 n=1,000，种子 11）加载到 fp16 底模上推理的 κ。
用法：110_quant_test.py <readout4|lora16> <gpu>。输出 pred_q_local_<mode>.csv（op_id, asa4, p1..p4）。"""
import os, sys, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
mode, gpu = sys.argv[1], int(sys.argv[2]); D = os.path.expanduser("~/research/NE_ERAS_JEV/data/derived"); M = os.path.expanduser("~/research/Anes_JEV_vs_LoRA/models")
c = pd.read_csv(f"{D}/asa_common_local.csv"); cal = c[c.split == "dev"].op_id.sample(1000, random_state=11); d = c[c.op_id.isin(list(c[c.split == "val"].op_id) + list(cal))].drop_duplicates("op_id").reset_index(drop=True)
if mode == "lora16": d = d[d.split == "val"].reset_index(drop=True)
path = "/mnt/c/backup/Qwen3.5-9B"; tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
P = "You are an anesthesiologist. Based on the preoperative information below, assign the ASA Physical Status class.\n\n%s\n\nOptions:\nA. ASA I\nB. ASA II\nC. ASA III\nD. ASA IV-V\nAnswer:"
if mode == "readout4":
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, dtype=torch.float16, device_map={"": gpu}).eval()
else:
    from peft import PeftModel
    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map={"": gpu}), f"{M}/LOC1000s11").eval()
a1 = [tok.encode(" " + x, add_special_tokens=False)[-1] for x in "ABCD"]; a2 = [tok.encode(x, add_special_tokens=False)[-1] for x in "ABCD"]
rows = []; t0 = time.time(); out = f"{D}/pred_q_local_{mode}.csv"; bs = 8 if mode == "lora16" else 1
for i in range(0, len(d), bs):
    b = d.iloc[i:i + bs]; x = tok([P % t for t in b.text], return_tensors="pt", padding=True).to(f"cuda:{gpu}")
    with torch.no_grad(): lg = m(**x, logits_to_keep=1).logits[:, -1].float()
    p = torch.softmax(lg[:, a1], -1).cpu().numpy() if mode == "lora16" else torch.softmax(torch.maximum(lg[:, a1], lg[:, a2]), -1).cpu().numpy()
    for r, q in zip(b.itertuples(), p): rows.append(dict(op_id=r.op_id, asa4=r.asa4, p1=q[0], p2=q[1], p3=q[2], p4=q[3]))
    if (i // bs + 1) % 500 == 0: print(i + bs, f"{(time.time()-t0)/(i+bs):.3f}s/例", flush=True)
pd.DataFrame(rows).to_csv(out, index=False); print("完成", len(rows), f"{time.time()-t0:.0f}s")
