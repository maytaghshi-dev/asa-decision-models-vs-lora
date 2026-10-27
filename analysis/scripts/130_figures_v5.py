"""v5 论文全部六张图（matplotlib 逐元素绘制；中英文各一版）。
只读取汇总结果，不读取任何患者级数据，因此可在公开仓库中直接复现：
  - analysis/reports/paired_differences.csv、review_numbers_105.json（图 4）
  - 图 3、5、6 的数字逐项抄自 analysis/reports/{calibration_curve,local_eval,arms_eval_v3,vitaldb_curve,guidance_report}.md，出处见各函数注释
输出 figures/{fig1_design,fig_flow,fig2_cost_efficiency,fig3_forest,fig4_vitaldb,fig5_guidance}_{en,zh}.{png,pdf,svg}
用法：python analysis/scripts/130_figures_v5.py"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R, OUT = f"{ROOT}/analysis/reports", f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)

# 统一配色：生成式 / 最小决策模型 / 校准决策模型 / OpenJev / LoRA / XGBoost
C = dict(gen="#a8442f", dm="#2e5e8c", cal="#17324f", oj="#c4831f", lora="#4b7a35", xgb="#808080",
         grey="#f3f3f3", edge="#6b6b6b", ink="#222222", soft_dm="#e9f0f7", soft_lora="#eef5ea", soft_gen="#f8ece8", soft_in="#fbf5e9")


def save(fig, name, lang):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{name}_{lang}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def card(ax, x, y, w, h, fc, ec, lw=1.1, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, x1, y1, x2, y2, color="#555555", ls="-", lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ 图 1 研究设计
def fig_design(lang):
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(14.0, 7.2)); ax.set_xlim(0, 140); ax.set_ylim(0, 70); ax.axis("off")
    X = dict(coh=1, inp=31, app=53, ev=113); W = dict(coh=26, inp=18, app=56, ev=26)
    H = (["数据来源", "输入", "比较的做法（同一底模 Qwen3.5-9B）", "评价"] if zh else
         ["Cohorts", "Inputs", "Approaches (one base model, Qwen3.5-9B)", "Evaluation"])
    for k, t in zip(("coh", "inp", "app", "ev"), H): ax.text(X[k], 67.0, t, fontsize=11, fontweight="bold", color=C["ink"])

    coh = ([("MOVER（美国）", "57,473 台 / 38,245 人", "开发数据源\n评价子集 4,000 台"),
            ("本院专病库\n（腹部肿瘤手术）", "7,079 台 / 6,804 人", "开发 2021–24：4,326 台\n时间外验证 2025–26：2,753 台"),
            ("VitalDB（韩国首尔）", "6,242 台 / 5,961 人", "第二站点\n训练 4,357 台 / 测试 1,885 台")] if zh else
           [("MOVER (USA)", "57,473 operations / 38,245 patients", "Development source\nevaluation subset 4,000"),
            ("Local specialty database\n(abdominal tumor surgery)", "7,079 operations / 6,804 patients", "Development 2021–24: 4,326\nTemporal validation 2025–26: 2,753"),
            ("VitalDB (Seoul, Korea)", "6,242 operations / 5,961 patients", "Second site\ntrain 4,357 / test 1,885")])
    ys = [50, 30, 10]; cx = X["coh"] + W["coh"] / 2
    for (t1, t2, t3), y in zip(coh, ys):
        card(ax, X["coh"], y, W["coh"], 14, C["grey"], C["edge"])
        ax.text(cx, y + 11.3, t1, ha="center", va="center", fontsize=8.4, fontweight="bold", linespacing=1.15)
        ax.text(cx, y + 7.4, t2, ha="center", va="center", fontsize=7.4)
        ax.text(cx, y + 3.4, t3, ha="center", va="center", fontsize=7.1, color="#444", linespacing=1.25)

    inp = ([("T1", "共有结构化变量\n年龄、性别、BMI、病史、\n血压心率、化验、手术名"), ("T2", "T1 + 术前文本\n本院：麻醉前访视原文\nVitalDB：诊断与检查")] if zh else
           [("T1", "Shared structured variables\nage, sex, BMI, history,\nvital signs, labs, procedure"), ("T2", "T1 + preoperative text\nlocal: preanesthesia note\nVitalDB: diagnoses, tests")])
    ix = X["inp"] + W["inp"] / 2
    for (tag, txt), y in zip(inp, (44, 18)):
        card(ax, X["inp"], y, W["inp"], 16, C["soft_in"], "#c99a45")
        ax.text(ix, y + 13.0, tag, ha="center", va="center", fontsize=10, fontweight="bold", color="#8a6420")
        ax.text(ix, y + 6.2, txt, ha="center", va="center", fontsize=7.0, linespacing=1.3)
    xr = X["coh"] + W["coh"] + 0.3; xl = X["inp"] - 0.3
    for y in ys: arrow(ax, xr, y + 7, xl, 52)
    arrow(ax, xr, 37, xl, 26); arrow(ax, xr, 17, xl, 26)
    arrow(ax, xr, 57, xl, 27.5, ls=(0, (3, 2)), color="#aaa")  # MOVER T2 仅探索性

    # 输入 → 做法：汇总线
    bx = X["inp"] + W["inp"] + 1.8
    ax.plot([X["inp"] + W["inp"], bx], [52, 52], color="#888", lw=1.1); ax.plot([X["inp"] + W["inp"], bx], [26, 26], color="#888", lw=1.1)
    ax.plot([bx, bx], [9.5, 59], color="#888", lw=1.1)
    for y in (59, 34.5, 9.5): arrow(ax, bx, y, X["app"] - 0.3, y, color="#888")

    A0, AW = X["app"], W["app"]
    card(ax, A0, 54, AW, 10, C["soft_gen"], C["gen"])
    ax.text(A0 + 2, 61.3, "生成式零样本提示" if zh else "Generative zero-shot prompting", fontsize=8.8, fontweight="bold", color=C["gen"], va="center")
    ax.text(A0 + 2, 57.0, ("先写不超过 3 句的理由再作答\n逐词解码，平均约 89 个输出词元" if zh else
                           "Writes a rationale of up to three sentences, then answers\ntoken-by-token decoding, about 89 output tokens per case"), fontsize=7.2, va="center", linespacing=1.3)
    card(ax, A0, 17, AW, 34, C["soft_dm"], C["dm"], lw=1.3)
    ax.text(A0 + 2, 48.4, ("开源非生成式决策模型（一次前向计算，不生成文本）" if zh else
                           "Open-source non-generative decision models (one forward pass, no text)"), fontsize=8.4, fontweight="bold", color=C["dm"], va="center")
    # 最小决策模型
    m0, mw = A0 + 2, 25
    card(ax, m0, 30.5, mw, 15, "white", C["dm"])
    ax.text(m0 + mw / 2, 43.2, "最小决策模型" if zh else "Minimal decision model", ha="center", va="center", fontsize=8.4, fontweight="bold", color=C["dm"])
    ax.text(m0 + mw / 2, 40.3, ("选项字母 A–D 的 logits → softmax" if zh else "softmax over option letters A–D"), ha="center", va="center", fontsize=6.9)
    for i, pv in enumerate([0.06, 0.58, 0.31, 0.05]):
        ax.add_patch(Rectangle((m0 + 3 + i * 4.2, 33.6), 2.8, pv * 8.5, fc=C["dm"] if i == 1 else "#9db5cf", ec="none"))
        ax.text(m0 + 4.4 + i * 4.2, 32.6, "ABCD"[i], ha="center", va="center", fontsize=6.4)
    ax.text(m0 + mw - 2.2, 35.0, "arg max", ha="center", va="center", fontsize=6.6, style="italic")
    # OpenJev
    o0 = A0 + 29
    card(ax, o0, 30.5, mw, 15, "white", C["oj"])
    ax.text(o0 + mw / 2, 42.5, "OpenJev", ha="center", va="center", fontsize=8.4, fontweight="bold", color=C["oj"])
    ax.text(o0 + mw / 2, 36.0, ("Jev 的开源兼容实现\n冻结配置，全部在院内运行" if zh else "Open implementation compatible\nwith Jev; frozen configuration;\nruns entirely on site"),
            ha="center", va="center", fontsize=6.9, linespacing=1.3)
    # 校准决策模型
    card(ax, A0 + 2, 18.5, AW - 4, 9.5, "white", C["cal"])
    ax.text(A0 + 4, 25.3, ("校准决策模型：在期望分 $E=\\Sigma_j\\, j\\,p_j$ 上，用 k 个本地标签拟合 3 个切点" if zh else
                           "Calibrated decision model: 3 cutpoints on $E=\\Sigma_j\\, j\\,p_j$, fitted on k local labels"), fontsize=7.4, fontweight="bold", color=C["cal"], va="center")
    l0, l1 = A0 + 6, A0 + AW - 6; xv = lambda v: l0 + (v - 1) / 3 * (l1 - l0)
    ax.plot([l0, l1], [21.3, 21.3], color="#444", lw=1)
    for v in (1, 2, 3, 4): ax.plot([xv(v)] * 2, [20.8, 21.8], color="#444", lw=1); ax.text(xv(v), 19.6, str(v), ha="center", va="center", fontsize=6.2)
    cps = [1.8, 2.2, 2.8]
    for c_ in cps: ax.plot([xv(c_)] * 2, [20.4, 22.9], color=C["gen"], lw=1.6)
    for lab, (a_, b_) in zip(("I", "II", "III", "IV–V"), zip([1] + cps, cps + [4])): ax.text(xv((a_ + b_) / 2), 22.7, lab, ha="center", va="center", fontsize=6.4, color=C["gen"])
    arrow(ax, m0 + mw / 2, 30.5, m0 + mw / 2, 28.2, color=C["cal"]); arrow(ax, o0 + mw / 2, 30.5, o0 + mw / 2, 28.2, color=C["cal"])
    # LoRA
    card(ax, A0, 4, AW, 11, C["soft_lora"], C["lora"])
    ax.text(A0 + 2, 11.6, "任务 LoRA（4-bit QLoRA）" if zh else "Task LoRA (4-bit QLoRA)", fontsize=8.8, fontweight="bold", color=C["lora"], va="center")
    ax.text(A0 + 2, 7.3, ("以 n 个本地标签训练（100 至全量，3 个种子）\n每个站点训练并保存一个适配器" if zh else
                          "Trained on n local labels (100 to the full set, three seeds)\none adapter trained and stored per site"), fontsize=7.2, va="center", linespacing=1.3)

    # 评价
    E0, EW = X["ev"], W["ev"]
    ev = ([("H1  优效", "最小决策模型\n> 生成式零样本"), ("H2  非劣效（界值 0.05）", "校准决策模型（1,000 标签）\n对 LoRA@1,000"), ("H3  优效", "全量 LoRA\n> 校准决策模型")] if zh else
          [("H1  superiority", "Minimal decision model\n> generative zero-shot"), ("H2  non-inferiority (0.05)", "Calibrated decision model\n(1,000 labels) vs. LoRA@1,000"), ("H3  superiority", "Full-set LoRA\n> calibrated decision model")])
    for i, (a_, b_) in enumerate(ev):
        y = 53 - i * 12.5
        card(ax, E0, y, EW, 10.5, "white", "#555")
        ax.text(E0 + 2, y + 8.0, a_, fontsize=8.0, fontweight="bold", va="center")
        ax.text(E0 + 2, y + 3.6, b_, fontsize=7.1, va="center", linespacing=1.25)
    card(ax, E0, 4, EW, 21, C["grey"], C["edge"])
    ax.text(E0 + 2, 22.2, "主要指标" if zh else "Primary metric", fontsize=7.9, fontweight="bold", va="center")
    ax.text(E0 + 2, 18.0, ("相对站点记录的二次加权 κ\n患者层面 cluster bootstrap" if zh else "Quadratic-weighted κ against\nthe site record; patient-level\ncluster bootstrap"), fontsize=6.9, va="center", linespacing=1.25)
    ax.text(E0 + 2, 12.2, "同时报告" if zh else "Also reported", fontsize=7.9, fontweight="bold", va="center")
    ax.text(E0 + 2, 8.2, ("延迟、词元、GPU 时间、\n所需标签数、ECE、矛盾内容" if zh else "latency, tokens, GPU time,\nlabels needed, ECE,\ncontradictory content"), fontsize=6.9, va="center", linespacing=1.25)
    ex = A0 + AW + 1.8
    for y in (59, 34.5, 9.5): ax.plot([A0 + AW, ex], [y, y], color="#888", lw=1.1)
    ax.plot([ex, ex], [9.5, 59], color="#888", lw=1.1); arrow(ax, ex, 34.5, E0 - 0.3, 34.5, color="#888")
    save(fig, "fig1_design", lang)


# ------------------------------------------------------------------ 图 2 纳入与划分流程
def fig_flow(lang):
    zh = lang == "zh"
    # 数字来自 analysis/reports/asa_dataset_v1.md、asa_mover_v1.md 与 VitalDB 构建记录
    cols = ["本院专病库" if zh else "Local specialty database", "MOVER", "VitalDB"]
    sub = (["腹部肿瘤手术，时长 > 1 h，不含内镜与宫腔镜手术", "美国加州大学尔湾分校", "首尔大学医院，非心脏手术"] if zh else
           ["Abdominal tumor surgery > 1 h; endoscopic and\nhysteroscopic procedures not included", "University of California, Irvine", "Seoul National University Hospital,\nnoncardiac surgery"])
    main = [[("手术麻醉概览", "8,852 台") if zh else ("Anesthesia overview", "8,852 operations"),
             ("患者信息表", "64,354 台") if zh else ("Patient table", "64,354 operations"),
             ("病例", "6,388 台") if zh else ("Cases", "6,388 operations")],
            [("有麻醉开始时间与 ASA", "7,348 台（6,828 人）") if zh else ("Anesthesia start and ASA recorded", "7,348 (6,828 patients)"),
             ("ASA I–V（去重）", "57,504 台") if zh else ("ASA I–V (deduplicated)", "57,504"),
             ("ASA I–V", "6,242 台（5,961 人）") if zh else ("ASA I–V", "6,242 (5,961 patients)")],
            [("年龄 ≥ 18 岁", "7,332 台") if zh else ("Age ≥ 18 y", "7,332"),
             ("年龄 ≥ 18 岁", "57,473 台（38,245 人）") if zh else ("Age ≥ 18 y", "57,473 (38,245 patients)"), None],
            [("术前 30 天内有麻醉前访视", "7,080 台") if zh else ("Preanesthesia visit within 30 d", "7,080"), None, None],
            [("分析集", "7,079 台（6,804 人）") if zh else ("Analysis set", "7,079 (6,804 patients)"),
             ("按患者随机划分", "") if zh else ("Random split by patient", ""), ("按患者随机划分", "") if zh else ("Random split by patient", "")]]
    excl = {(0, 1): "缺开始时间或 ASA：1,504" if zh else "No start time or ASA: 1,504",
            (1, 1): "缺 ASA 等：6,850" if zh else "No ASA, other: 6,850",
            (2, 1): "ASA 缺失或 6 级：146" if zh else "ASA missing or 6: 146",
            (0, 2): "< 18 岁：16" if zh else "< 18 y: 16", (1, 2): "< 18 岁：31" if zh else "< 18 y: 31",
            (0, 3): "无术前 30 天内访视：252" if zh else "No visit within 30 d: 252",
            (0, 4): "跨期再手术：1" if zh else "Cross-period re-operation: 1"}
    final = [[(("开发集 2021–24", "4,326 台") if zh else ("Development 2021–24", "4,326")), (("时间外验证集 2025–26", "2,753 台（2,640 人）") if zh else ("Temporal validation 2025–26", "2,753 (2,640 patients)"))],
             [(("开发 80%", "45,994 台") if zh else ("Development 80%", "45,994")), (("测试 20%：11,479 台", "评价子集 4,000 台（2,606 人）") if zh else ("Test 20%: 11,479", "Evaluation subset 4,000 (2,606 patients)"))],
             [(("训练 70%", "4,357 台") if zh else ("Training 70%", "4,357")), (("测试 30%", "1,885 台（1,788 人）") if zh else ("Test 30%", "1,885 (1,788 patients)"))]]
    fig, ax = plt.subplots(figsize=(13.8, 9.2)); ax.set_xlim(0, 137); ax.set_ylim(0, 92); ax.axis("off")
    xs = [3, 48, 93]; w, h = 24, 7.0; ys = [76, 64, 52, 40, 28]
    for ci, x in enumerate(xs):
        ax.text(x + w / 2 + 6, 90, cols[ci], ha="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2 + 6, 85.6, sub[ci], ha="center", va="center", fontsize=7.4, color="#555", linespacing=1.25)
        last = None
        for ri, row in enumerate(main):
            b = row[ci]
            if b is None: continue
            y = ys[ri]
            card(ax, x, y, w, h, "white", C["dm"])
            ax.text(x + w / 2, y + (h * 0.66 if b[1] else h / 2), b[0], ha="center", va="center", fontsize=7.8, fontweight="bold")
            if b[1]: ax.text(x + w / 2, y + h * 0.28, b[1], ha="center", va="center", fontsize=7.6)
            if last is not None: arrow(ax, x + w / 2, ys[last], x + w / 2, y + h)
            if (ci, ri) in excl:
                ym = (ys[last] + y + h) / 2 if last is not None else y
                ax.plot([x + w / 2, x + w + 2], [ym, ym], color="#999", lw=0.9)
                card(ax, x + w + 2, ym - 2.4, 16.5, 4.8, "#fafafa", "#b0b0b0", lw=0.8, r=0.5)
                ax.text(x + w + 10.25, ym, excl[(ci, ri)], ha="center", va="center", fontsize=6.6, color="#555")
            last = ri
        # 末端两个集合：并列放置，由分析集分叉
        cx = x + w / 2; xl, xr, bw = x - 1, x + 21, 19.5
        ax.plot([cx, cx], [ys[4], 24.5], color="#555", lw=1.1); ax.plot([xl + bw / 2, xr + bw / 2], [24.5, 24.5], color="#555", lw=1.1)
        for j, (t1, t2) in enumerate(final[ci]):
            bx = (xl, xr)[j]
            arrow(ax, bx + bw / 2, 24.5, bx + bw / 2, 21.0)
            card(ax, bx, 11.5, bw, 9.5, C["soft_lora"], C["lora"])
            ax.text(bx + bw / 2, 18.0, t1, ha="center", va="center", fontsize=7.4, fontweight="bold")
            ax.text(bx + bw / 2, 14.2, t2.replace("（", "\n（").replace(" (", "\n(") if len(t2) > 18 else t2, ha="center", va="center", fontsize=7.0, linespacing=1.2)
    ax.text(68.5, 7.5, ("本院开发集与验证集之间、MOVER 与 VitalDB 各划分之间均无重叠患者" if zh else
                      "No patient appears in more than one set within any data source"), ha="center", fontsize=7.4, color="#555")
    ax.set_ylim(5, 92)
    save(fig, "fig_flow", lang)


# ------------------------------------------------------------------ 图 3 速度、精度与标签效率
def fig_efficiency(lang):
    zh = lang == "zh"
    # 数字：local_eval.md、arms_eval_v3.md、calibration_curve.md（曲线脚本粗网格）、表 2 延迟
    t = (dict(a="速度与精度（本院验证集）", x="单例推理延迟中位数（ms，批量 1，对数轴）", y="二次加权 κ",
              k="本地标签数（对数轴；最左点 k = 0 为未校准）", gen="生成式零样本", dm0="最小决策模型，零样本",
              c100="校准决策模型，100 个标签", c1k="校准决策模型，1,000 个标签", l1k="LoRA，1,000 个标签", lf="LoRA，全量",
              ck="校准决策模型（k 个标签，无训练）", lk="任务 LoRA（n 个标签训练）", xg="XGBoost", loc="本院验证集", mov="MOVER 测试子集",
              note="阴影：校准 2.5–97.5 百分位；误差线：LoRA 种子范围") if zh else
         dict(a="Speed versus accuracy (local validation set)", x="Median per-case latency (ms, batch 1, log scale)", y="Quadratic-weighted κ",
              k="Local labels (log scale; leftmost point k = 0 is uncalibrated)", gen="Generative zero-shot", dm0="Minimal decision model, zero-shot",
              c100="Calibrated decision model, 100 labels", c1k="Calibrated decision model, 1,000 labels", l1k="LoRA, 1,000 labels", lf="LoRA, full set",
              ck="Calibrated decision model (k labels, no training)", lk="Task LoRA (trained on n labels)", xg="XGBoost", loc="Local validation set", mov="MOVER test subset",
              note="Shading: calibration 2.5–97.5th percentile; bars: LoRA seed range"))
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.3))
    pts = [(t["gen"], 7067, 0.227, C["gen"], "o"), (t["dm0"], 111, 0.443, C["dm"], "o"), (t["c100"], 111, 0.526, "#4a7fb5", "D"),
           (t["c1k"], 111, 0.534, C["cal"], "D"), (t["l1k"], 179, 0.559, C["lora"], "s"), (t["lf"], 179, 0.561, "#2f5220", "s")]
    for n, x, y, c, m in pts: ax[0].scatter(x, y, s=58, color=c, marker=m, zorder=3, label=f"{n}  (κ {y:.3f})")
    ax[0].set_xscale("log"); ax[0].set_xlim(70, 12000); ax[0].set_ylim(0.15, 0.62)
    ax[0].set_xlabel(t["x"]); ax[0].set_ylabel(t["y"]); ax[0].set_title(t["a"], fontsize=10); ax[0].grid(alpha=.25)
    ax[0].legend(fontsize=6.8, loc="upper right", frameon=False)

    def curve(a, title, ks, km, klo, khi, ln, lm, llo, lhi, xn, xm):
        a.fill_between([1] + ks, klo, khi, color=C["dm"], alpha=.14, lw=0)
        a.plot([1] + ks, km, "D-", color=C["cal"], ms=3.8, label=t["ck"])
        a.errorbar(ln, lm, yerr=[np.array(lm) - np.array(llo), np.array(lhi) - np.array(lm)], fmt="s-", color=C["lora"], capsize=2.5, ms=4, label=t["lk"])
        a.plot(xn, xm, "^--", color=C["xgb"], ms=4, label=t["xg"])
        a.set_xscale("log"); a.set_xlabel(t["k"]); a.set_ylabel(t["y"]); a.set_title(title, fontsize=10); a.grid(alpha=.25); a.set_ylim(0.2, 0.62)
    curve(ax[1], t["loc"], [25, 50, 100, 200, 500, 1000], [0.443, 0.491, 0.516, 0.526, 0.528, 0.535, 0.540],
          [0.443, 0.384, 0.471, 0.474, 0.501, 0.520, 0.540], [0.443, 0.546, 0.547, 0.547, 0.544, 0.540, 0.540],
          [100, 300, 1000, 4326], [0.423, 0.542, 0.559, 0.561], [0.338, 0.530, 0.553, 0.556], [0.469, 0.556, 0.563, 0.570],
          [100, 300, 1000, 4326], [0.431, 0.467, 0.521, 0.549])
    curve(ax[2], t["mov"], [25, 50, 100, 200, 500, 1000], [0.473, 0.467, 0.495, 0.508, 0.517, 0.521, 0.524],
          [0.473, 0.372, 0.422, 0.476, 0.495, 0.516, 0.524], [0.473, 0.519, 0.530, 0.528, 0.529, 0.529, 0.524],
          [100, 300, 1000, 3000, 12000], [0.410, 0.411, 0.505, 0.563, 0.583], [0.394, 0.256, 0.462, 0.551, 0.580], [0.438, 0.500, 0.527, 0.575, 0.587],
          [100, 300, 1000, 3000, 12000], [0.234, 0.305, 0.395, 0.469, 0.504])
    h, l = ax[2].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.66, -0.07))
    ax[1].text(0.02, 0.03, t["note"], transform=ax[1].transAxes, fontsize=6.5, color="#555")
    for a, s in zip(ax, "ABC"): a.text(-0.13, 1.06, s, transform=a.transAxes, fontsize=12, fontweight="bold")
    plt.tight_layout(); save(fig, "fig2_cost_efficiency", lang)


# ------------------------------------------------------------------ 图 4 配对差森林图
def fig_forest(lang):
    zh = lang == "zh"
    P = pd.read_csv(f"{R}/paired_differences.csv")
    def g(ds, hyp, contrast):
        r = P[(P.dataset == ds) & (P.hyp == hyp) & (P.contrast == contrast)]
        return None if r.empty else (float(r.iloc[0]["diff"]), float(r.iloc[0]["lo"]), float(r.iloc[0]["hi"]))
    DM, OJ = C["dm"], C["oj"]
    spec = [  # 组, 中文标签, 英文标签, 数据集, 假设, 对比（结果文件中的键）, 颜色
        ("H1", "最小决策模型 − 生成式（本院）", "Minimal decision model − generative (local)", "本院 val（生成式子集）", "H1", "读出 T1 未校准 − 生成式零样本", DM),
        ("H1", "OpenJev 校准 − 生成式（本院）", "OpenJev, calibrated − generative (local)", "本院 val（生成式子集）", "H1", "OpenJev T1 校准 − 生成式零样本", OJ),
        ("H1", "最小决策模型 − 生成式（MOVER）", "Minimal decision model − generative (MOVER)", "MOVER 子集（生成式子集）", "H1", "读出 T1 未校准 − 生成式零样本", DM),
        ("H1", "OpenJev 校准 − 生成式（MOVER）", "OpenJev, calibrated − generative (MOVER)", "MOVER 子集（生成式子集）", "H1", "OpenJev T1 校准 − 生成式零样本", OJ),
        ("H2", "决策模型 T1，本院（三种子均值）", "Decision model T1, local (three-seed mean)", "本院 val", "H2", "读出 T1 校准(1000) − LoRA@1000 三种子均值", DM),
        ("H2", "决策模型 T1，本院（预先指定种子）", "Decision model T1, local (prespecified seed)", "本院 val", "H2", "读出 T1 校准(1000) − LoRA@1000 种子20261017", DM),
        ("H2", "决策模型 T1，本院（种子 11，标签匹配）", "Decision model T1, local (seed 11, label-matched)", "本院 val", "H2", "读出 T1 校准(1000) − LoRA@1000 种子11", DM),
        ("H2", "OpenJev T1，本院（三种子均值）", "OpenJev T1, local (three-seed mean)", "本院 val", "H2", "OpenJev T1 校准(1000) − LoRA@1000 三种子均值", OJ),
        ("H2", "决策模型 T2，本院（单种子）", "Decision model T2, local (single seed)", "本院 val", "H2", "读出 T2 校准(1000) − LoRA T2@1000（单种子）", DM),
        ("H2", "OpenJev T2，本院（单种子）", "OpenJev T2, local (single seed)", "本院 val", "H2", "OpenJev T2 校准(1000) − LoRA T2@1000（单种子）", OJ),
        ("H2", "决策模型 T1，MOVER（三种子均值）", "Decision model T1, MOVER (three-seed mean)", "MOVER 子集", "H2", "读出 T1 校准(1000) − LoRA@1000 三种子均值", DM),
        ("H2", "OpenJev T1，MOVER（三种子均值）", "OpenJev T1, MOVER (three-seed mean)", "MOVER 子集", "H2", "OpenJev T1 校准(1000) − LoRA@1000 三种子均值", OJ),
        ("H2", "决策模型 T2，VitalDB（三种子均值）", "Decision model T2, VitalDB (three-seed mean)", "VitalDB", "H2", "读出 T2 校准(1000) − 目标站点 LoRA T2@1000（种子均值）", DM),
        ("H3", "LoRA 全量 − 决策模型 T1，本院", "Full LoRA − decision model T1, local", "本院 val", "H3", "LoRA@全量 三种子均值 − 读出 T1 校准(1000)", DM),
        ("H3", "LoRA 全量 − OpenJev T1，本院", "Full LoRA − OpenJev T1, local", "本院 val", "H3", "LoRA@全量 三种子均值 − OpenJev T1 校准(1000)", OJ),
        ("H3", "LoRA T2 全量 − 决策模型 T2，本院", "Full LoRA − decision model T2, local", "本院 val", "H3", "LoRA T2@全量（单种子）− 读出 T2 校准(1000)", DM),
        ("H3", "LoRA T2 全量 − OpenJev T2，本院", "Full LoRA − OpenJev T2, local", "本院 val", "H3", "LoRA T2@全量（单种子）− OpenJev T2 校准(1000)", OJ),
        ("H3", "LoRA@3,000 − 决策模型 T1，MOVER", "LoRA@3,000 − decision model T1, MOVER", "MOVER 子集", "H3", "LoRA@3000 三种子均值 − 读出 T1 校准(1000)", DM),
        ("H3", "LoRA@12,000 − 决策模型 T1，MOVER", "LoRA@12,000 − decision model T1, MOVER", "MOVER 子集", "H3", "LoRA@12000 三种子均值 − 读出 T1 校准(1000)", DM),
        ("H3", "LoRA@3,000 − 决策模型 T2，VitalDB", "LoRA@3,000 − decision model T2, VitalDB", "VitalDB", "H3", "目标站点 LoRA T2@3000（种子均值）− 读出 T2 校准(1000)", DM)]
    rows = [(grp, zl if zh else el, g(d, h, c), col) for grp, zl, el, d, h, c, col in spec]
    rows = [r for r in rows if r[2] is not None]
    N = json.load(open(f"{R}/review_numbers_105.json"))  # 含标签抽样与种子变异的 H2（50 个标签池 × 3 个 LoRA 种子）
    k = max(i for i, r in enumerate(rows) if r[0] == "H2" and ("11" in r[1]))
    rows.insert(k + 1, ("H2", "决策模型 T1，本院（50 个标签池）" if zh else "Decision model T1, local (50 label pools)", (N["comb_diff"], N["comb_lo"], N["comb_hi"]), DM))
    ttl = dict(H1=("H1  决策模型 − 生成式零样本（优效）", "H1  Decision model − generative zero-shot (superiority)"),
               H2=("H2  校准决策模型（1,000 标签）− LoRA@1,000（非劣效，界值 −0.05）", "H2  Calibrated decision model (1,000 labels) − LoRA@1,000 (non-inferiority, margin −0.05)"),
               H3=("H3  LoRA − 校准决策模型（1,000 标签）（优效）", "H3  LoRA − calibrated decision model (1,000 labels) (superiority)"))
    cnt = [sum(r[0] == kk for r in rows) for kk in ("H1", "H2", "H3")]
    fig, axs = plt.subplots(3, 1, figsize=(8.6, 0.34 * len(rows) + 2.6), gridspec_kw=dict(height_ratios=cnt, hspace=0.6))
    for ax, kk in zip(axs, ("H1", "H2", "H3")):
        rr = [r for r in rows if r[0] == kk]
        for i, (_, lab, (d, lo, hi), col) in enumerate(rr):
            y = len(rr) - 1 - i; ax.plot([lo, hi], [y, y], color=col, lw=1.8); ax.scatter(d, y, color=col, s=26, zorder=3)
            fz = lambda v: "0.000" if abs(v) < 0.0005 else f"{v:+.3f}"
            ax.text(1.02, y, f"{fz(d)} ({fz(lo)}, {fz(hi)})", transform=ax.get_yaxis_transform(), va="center", fontsize=7.4)
        ax.set_yticks(range(len(rr))); ax.set_yticklabels([r[1] for r in rr][::-1], fontsize=7.6); ax.axvline(0, color="#333", lw=0.8)
        if kk == "H2": ax.axvline(-0.05, color=C["gen"], ls="--", lw=1)
        ax.set_title(ttl[kk][0 if zh else 1], fontsize=8.8, loc="left", fontweight="bold"); ax.set_ylim(-0.7, len(rr) - 0.3); ax.grid(axis="x", alpha=.25)
        ax.set_xlabel("κ 差值（95% CI）" if zh else "κ difference (95% CI)", fontsize=8)
    save(fig, "fig3_forest", lang)


# ------------------------------------------------------------------ 图 5 VitalDB 目标站点标签效率
def fig_vitaldb(lang):
    zh = lang == "zh"
    # 数字：analysis/reports/vitaldb_curve.md（k = 0 为未校准；1,000 标签只有一个池）
    ks = [1, 25, 50, 100, 200, 500, 1000]
    dm = ([0.380, 0.509, 0.533, 0.549, 0.555, 0.563, 0.567], [0.380, 0.395, 0.473, 0.491, 0.515, 0.554, 0.567], [0.380, 0.564, 0.566, 0.569, 0.569, 0.568, 0.567])
    oj = ([0.370, 0.509, 0.531, 0.554, 0.562, 0.572, 0.582], [0.370, 0.397, 0.474, 0.515, 0.529, 0.550, 0.582], [0.370, 0.570, 0.573, 0.581, 0.583, 0.584, 0.582])
    ln, lm, ll, lh = [100, 300, 1000, 3000], [0.282, 0.470, 0.567, 0.584], [0.061, 0.365, 0.532, 0.561], [0.444, 0.570, 0.602, 0.599]
    t = (("VitalDB 测试集（T2，n = 1,885）", "目标站点标签数（对数轴；最左点 k = 0 为未校准）", "二次加权 κ", "校准决策模型（k 个标签）", "OpenJev，k 个标签校准", "目标站点 LoRA（种子均值，误差线为范围）") if zh else
         ("VitalDB test set (T2, n = 1,885)", "Target-site labels (log scale; leftmost point k = 0 is uncalibrated)", "Quadratic-weighted κ", "Calibrated decision model (k labels)", "OpenJev, calibrated with k labels", "Target-site LoRA (seed mean; bars = range)"))
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for (m, lo, hi), col, lab, mk in ((dm, C["cal"], t[3], "D"), (oj, C["oj"], t[4], "o")):
        ax.fill_between(ks, lo, hi, color=col, alpha=.12, lw=0); ax.plot(ks, m, mk + "-", color=col, ms=3.8, label=lab)
    ax.errorbar(ln, lm, yerr=[np.array(lm) - np.array(ll), np.array(lh) - np.array(lm)], fmt="s-", color=C["lora"], capsize=2.5, ms=4, label=t[5])
    ax.set_xscale("log"); ax.set_xlabel(t[1]); ax.set_ylabel(t[2]); ax.set_title(t[0], fontsize=10); ax.grid(alpha=.25); ax.set_ylim(0, 0.65)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    plt.tight_layout(); save(fig, "fig4_vitaldb", lang)


# ------------------------------------------------------------------ 图 6 自然语言指导
def fig_guidance(lang):
    zh = lang == "zh"
    # 数字：analysis/reports/guidance_report.md；参考线：1,000 标签校准 T1 0.534 / T2 0.545，全量本院 LoRA T1 0.561（三种子均值）/ T2 0.606
    orig = [1.3, 47.0, 48.8, 2.9]
    mix = {"t1": [[11.1, 59.1, 27.4, 2.4], [43.1, 35.7, 20.2, 1.1], [0.0, 85.2, 14.6, 0.2], [64.8, 24.2, 10.5, 0.6]],
           "t2": [[10.4, 20.1, 59.5, 10.0], [8.7, 13.0, 70.4, 7.9], [0.0, 45.3, 50.6, 4.1], [18.4, 35.2, 38.9, 7.4]]}
    kap = {"t1": [0.443, 0.312, 0.272, 0.187], "t2": [0.505, 0.435, 0.557, 0.528]}
    cal, lora = {"t1": 0.534, "t2": 0.545}, {"t1": 0.561, "t2": 0.606}
    nm = (["原始记录", "无指导", "ASA 2020 官方定义", "本地记录习惯", "严格遵循指南"] if zh else ["Recorded ASA", "No guidance", "ASA 2020 definitions", "Local recording habit", "Strict guideline"])
    CL = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.1), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    for a, tv in zip(ax[:2], ("t1", "t2")):
        dist = [orig] + mix[tv]; bot = np.zeros(5)
        for k in range(4):
            v = np.array([x[k] for x in dist]); a.barh(range(5), v, left=bot, color=CL[k], label=["I", "II", "III", "IV–V"][k], height=0.68); bot += v
        a.set_yticks(range(5)); a.set_yticklabels(nm); a.invert_yaxis(); a.set_xlim(0, 100); a.set_xlabel("%")
        a.set_title(tv.upper() + ("：预测类别构成" if zh else ": predicted class mix"), fontsize=10)
        for i, kv in enumerate(kap[tv]): a.text(101.5, i + 1, f"κ {kv:.3f}", va="center", fontsize=7.8)
    ax[0].legend(ncol=4, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    w = 0.38
    for j, (tv, col) in enumerate((("t1", "#9ecae1"), ("t2", "#2171b5"))):
        ax[2].bar(np.arange(4) + (j - 0.5) * w, kap[tv], w, color=col, label=tv.upper())
        ax[2].hlines(cal[tv], -0.5, 3.5, color=col, ls="--", lw=1); ax[2].hlines(lora[tv], -0.5, 3.5, color=col, ls=":", lw=1.4)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(nm[1:], rotation=20, fontsize=7.8); ax[2].set_ylabel("κ"); ax[2].set_ylim(0, 0.75)
    ax[2].legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    ax[2].set_title(("虚线：校准决策模型（1,000 标签）；点线：全量本院 LoRA" if zh else "Dashed: calibrated decision model (1,000 labels); dotted: full local LoRA"), fontsize=7.8)
    for a, s in zip(ax, "ABC"): a.text(-0.1, 1.05, s, transform=a.transAxes, fontsize=12, fontweight="bold")
    plt.tight_layout(); save(fig, "fig5_guidance", lang)


if __name__ == "__main__":
    for lang in ("en", "zh"):
        fig_design(lang); fig_flow(lang); fig_efficiency(lang); fig_forest(lang); fig_vitaldb(lang); fig_guidance(lang)
    print("figures written to", OUT)
