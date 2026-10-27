#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OSF 预注册：建草稿 / 取 schema / 写入答案 / 回读校验（Anes_JEV_vs_LoRA）。

用法：
  python osf_lora_register.py schema                      # 列出可用注册模板（拿 schema id）
  python osf_lora_register.py schema-questions <schema>   # 导出该模板的 qid + 题干 → JSON
  python osf_lora_register.py create <schema>             # 建草稿，打印 draft id
  python osf_lora_register.py fill <draft_id> <answers.json>   # 扁平 responses 写入
  python osf_lora_register.py read <draft_id>             # 回读并逐栏对比 answers.json

认证：~/.osf_browser_state.json（浏览器 storage state 的 cookies）。
坑（已固化）：答案只走扁平 registration_responses；file 字段传 []；服务端 HTML 转义一次，
读写都要先 html.unescape；PUT/嵌套 metadata 一律 502。
"""
import json, sys, os, html
import requests

BASE = "https://api.osf.io/v2"
STATE = os.path.expanduser("~/.osf_browser_state.json")
PROXY = os.environ.get("OSF_PROXY", "http://127.0.0.1:7890")


def session():
    st = json.load(open(STATE))
    s = requests.Session()
    for c in st.get("cookies", []):
        if "osf.io" in c.get("domain", ""):
            s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))
    s.headers.update({"Accept": "application/vnd.api+json", "Content-Type": "application/vnd.api+json"})
    s.proxies = {"http": PROXY, "https": PROXY}
    return s


def call(method, path, **kw):
    s = session()
    try:
        r = s.request(method, BASE + path, timeout=45, **kw)
    except Exception as e:
        raise SystemExit(f"{method} {path} -> 连接失败（{type(e).__name__}）: {str(e)[:200]}\n"
                         f"先确认网络/代理（OSF_PROXY={PROXY}）；api.osf.io 宕机时所有端点返回 502。")
    if r.status_code >= 400:
        hint = "（OSF 侧故障：查 https://status.cos.io）" if r.status_code in (502, 503, 504) else \
               "（401/403 = cookie 失效，需刷新 ~/.osf_browser_state.json）" if r.status_code in (401, 403) else ""
        raise SystemExit(f"{method} {path} -> {r.status_code}{hint}\n{r.text[:600]}")
    return r.json() if r.content else {}


def cmd_schema():
    d = call("GET", "/schemas/registrations/?page[size]=100")
    for x in d["data"]:
        name = x["attributes"].get("name")
        print(f'{x["id"]}  v{x["attributes"].get("version")}  {name}')


def cmd_schema_questions(schema_id):
    d = call("GET", f"/schemas/registrations/{schema_id}/")
    schema = d["data"]["attributes"].get("schema") or {}
    out = {}

    def walk(node, page=None):
        if isinstance(node, dict):
            if node.get("qid"):
                out[node["qid"]] = {
                    "title": node.get("title"),
                    "description": (node.get("description") or "")[:300],
                    "type": node.get("type"),
                    "page": page,
                }
            for k, v in node.items():
                walk(v, node.get("title") if k == "pages" else page)
        elif isinstance(node, list):
            for v in node:
                walk(v, page)

    walk(schema)
    p = f"osf_schema_{schema_id}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"{len(out)} 个字段 → {p}")
    for q, v in out.items():
        print(f'{q:10s} [{str(v["type"]):12s}] {str(v["title"])[:80]}')


def cmd_create(schema_id):
    body = {"data": {"type": "draft_registrations",
                     "attributes": {"registration_responses": {}},
                     "relationships": {"registration_schema": {"data": {"type": "registration_schemas", "id": schema_id}}}}}
    d = call("POST", "/draft_registrations/", data=json.dumps(body))
    did = d["data"]["id"]
    print(f"draft id = {did}")
    print(f"editor    = https://osf.io/registries/drafts/{did}/edit")


def cmd_fill(draft_id, answers_path):
    ans = json.load(open(answers_path))
    if "responses" in ans:
        ans = ans["responses"]
    ans = {k: (html.unescape(v) if isinstance(v, str) else v) for k, v in ans.items()}
    call("PATCH", f"/draft_registrations/{draft_id}/",
         data=json.dumps({"data": {"type": "draft_registrations", "id": draft_id,
                                   "attributes": {"registration_responses": ans}}}))
    print(f"已写入 {len(ans)} 栏 → 回读校验：")
    cmd_read(draft_id, answers_path)


def cmd_read(draft_id, answers_path=None):
    d = call("GET", f"/draft_registrations/{draft_id}/")["data"]
    got = d["attributes"].get("registration_responses") or {}
    print(f"草稿 {draft_id}: {len(got)} 栏有内容")
    if answers_path:
        ans = json.load(open(answers_path))
        if "responses" in ans:
            ans = ans["responses"]
        bad = 0
        for k, v in ans.items():
            have = got.get(k, "")
            have_u = html.unescape(have) if isinstance(have, str) else have
            if have_u != v:
                bad += 1
                print(f"  ✗ {k}: 发送 {str(v)[:60]!r} / 回读 {str(have_u)[:60]!r}")
        print("  全部一致" if bad == 0 else f"  {bad} 栏不一致")
    else:
        for k, v in sorted(got.items()):
            print(f"  {k}: {str(html.unescape(v) if isinstance(v, str) else v)[:90]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    c = sys.argv[1]
    if c == "schema":
        cmd_schema()
    elif c == "schema-questions":
        cmd_schema_questions(sys.argv[2])
    elif c == "create":
        cmd_create(sys.argv[2])
    elif c == "fill":
        cmd_fill(sys.argv[2], sys.argv[3])
    elif c == "read":
        cmd_read(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        raise SystemExit(__doc__)
