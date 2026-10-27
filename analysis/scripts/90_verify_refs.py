"""参考文献核对（AI4Scholar API，批量取 title/year/authors/venue/externalIds），与文献文件 #1–#32 的记录比对：标题词重合、第一作者姓、年份、DOI/PMID。
输出 reports/refs_verification.md 与 reports/refs_verification.json（含 API 返回原文，便于复核）。未能命中的条目标注，留给 metstr/PubMed 兜底。"""
import os, re, sys, json, unicodedata
sys.path.insert(0, os.path.expanduser("~/research-orchestrator/scripts"))
import ai4scholar_client as A
R = os.path.expanduser("~/research/Anes_JEV_vs_LoRA/analysis/reports"); T = open(os.path.expanduser("~/research/Anes_JEV_vs_LoRA/literature/2026-10-16_JEV_vs_LoRA_anesthesia_lit.md"), encoding="utf-8").read()
rows = {}
for l in T.split("\n"):
    if re.match(r"\|\s*\d+\s*\|", l):
        c = [x.strip() for x in l.strip("|").split("|")]; rows[int(c[0])] = c[1]
def ids(text):
    out = []
    for m in re.findall(r"PMID[ :]*(\d{6,9})", text): out.append(("PMID:" + m, m))
    for m in re.findall(r"(?:doi:|doi\s*)(10\.\d{4,9}/[^\s；;，,）)（(]+)", text, flags=re.I): out.append(("DOI:" + m.rstrip(".。"), m))
    for m in re.findall(r"arXiv:(\d{4}\.\d{4,5})", text): out.append(("ARXIV:" + m, m))
    return out
req = {}; order = []
for k, t in rows.items():
    for pid, raw in ids(t):
        req.setdefault(pid, []).append(k)
        if pid not in order: order.append(pid)
print(len(rows), "条；待查标识", len(order), flush=True)
fields = "title,year,authors,journal,venue,externalIds,publicationDate"
res = {}
import time
for pid in order:
    for attempt in range(3):
        try: res[pid] = A.request_json(f"/graph/v1/paper/{pid}", params={"fields": fields}); break
        except Exception as e:
            res[pid] = None; err = str(e)[:80]; time.sleep(1.5)
    print(pid, "OK" if res[pid] else "失败 " + err, flush=True)
norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", unicodedata.normalize("NFKD", (s or "").lower()))
def words(s): return set(w for w in norm(s).split() if len(w) > 3)
out = ["# 参考文献核对（AI4Scholar API；对照 literature/2026-10-16_JEV_vs_LoRA_anesthesia_lit.md）\n", "判定：**命中**=标识解析到论文且标题词重合≥0.5；**弱匹配**=解析到但重合低或第一作者姓不符，需人工看；**未命中**=API 无记录。\n", "| # | 标识 | API 标题 | API 年/刊 | 重合 | 判定 |", "|---|---|---|---|---|---|"]
verdict = {}
for k, t in rows.items():
    cand = ids(t)
    if not cand: verdict[k] = "无标识"; out.append(f"| {k} | — | — | — | — | 无标识（待补）|"); continue
    best = None
    for pid, raw in cand:
        d = res.get(pid)
        if d and d.get("title"):
            ov = len(words(t) & words(d["title"])) / max(1, len(words(d["title"]))); au = (d.get("authors") or [{}])[0].get("name", "").split(" ")[-1].lower()
            sc = ov + (0.2 if au and au in norm(t) else 0)
            if best is None or sc > best[0]: best = (sc, pid, d, ov, au)
    if not best: verdict[k] = "未命中"; out.append(f"| {k} | {cand[0][0]} | — | — | — | 未命中 |"); continue
    sc, pid, d, ov, au = best; v = "命中" if ov >= 0.5 else "弱匹配"; verdict[k] = v
    out.append(f"| {k} | {pid} | {d['title'][:90]} | {d.get('year')} / {(d.get('journal') or {}).get('name') or d.get('venue') or ''} | {ov:.2f} | {v} |")
cnt = {v: list(verdict.values()).count(v) for v in set(verdict.values())}; out.append(f"\n汇总：{cnt}")
open(f"{R}/refs_verification.md", "w").write("\n".join(out)); json.dump({"verdict": verdict, "raw": res}, open(f"{R}/refs_verification.json", "w"), ensure_ascii=False, indent=1); print("\n".join(out))
