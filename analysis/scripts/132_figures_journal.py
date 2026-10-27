"""Journal-style redraw of all six manuscript figures (Nature-portfolio conventions).

Design rules, applied to every figure:
  * drawn at final print size (double column 165 mm, single column 85 mm), so text is never shrunk afterwards;
  * one type size (7 pt) for all text, Helvetica-type sans for English, Song-type serif for Chinese; bold only for panel labels;
  * colour-blind-safe Okabe-Ito accents limited to three hues plus grey;
  * square, unfilled boxes with 0.5 pt rules, no rounded corners, no shadows or fills;
  * direct labelling instead of large legends where possible; panels labelled a), b), c).
Reads aggregate results only (same sources as 130_figures_v5.py), no patient-level data.
Output: figures/{fig1_design,fig_flow,fig2_cost_efficiency,fig3_forest,fig4_vitaldb,fig5_guidance}_{en,zh}.{png,pdf,svg}"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

for f in ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(f): fm.fontManager.addfont(f)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R, OUT = f"{ROOT}/analysis/reports", f"{ROOT}/figures"
MM = 1 / 25.4; W2, W1 = 165 * MM, 85 * MM
BLUE, SKY, VERM, AMBER, GREY, INK = "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#7A7A7A", "#1A1A1A"
FS = 7

def style(lang):
    fam = ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"] if lang == "en" else ["Noto Serif CJK SC", "DejaVu Serif"]
    plt.rcParams.update({"font.family": fam, "font.sans-serif": fam, "font.serif": fam,
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False, "legend.frameon": False,
        "axes.titleweight": "normal", "pdf.fonttype": 42 if lang == "en" else 3, "svg.fonttype": "none", "mathtext.fontset": "custom",
        "mathtext.rm": fam[0], "mathtext.it": fam[0] + ":italic"})

def save(fig, name, lang):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{name}_{lang}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def panel(ax, s, x=-0.02, y=1.02, trans=None):
    ax.text(x, y, s, transform=trans or ax.transAxes, fontsize=FS + 1, fontweight="bold", va="bottom", ha="left")

def box(ax, x, y, w, h, lw=0.5, ec=INK):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=lw, ec=ec))

def arrow(ax, x1, y1, x2, y2, lw=0.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", lw=lw, color=INK, shrinkA=0, shrinkB=0))

T = lambda lang, en, zh: en if lang == "en" else zh


# ================================================================ Fig. 1 study design
def fig_design(lang):
    style(lang); fig = plt.figure(figsize=(W2, 112 * MM))
    # ---- a) cohorts as a ruled table
    ax = fig.add_axes([0, 0.64, 1, 0.33]); ax.set_xlim(0, 165); ax.set_ylim(0, 36); ax.axis("off"); panel(ax, "a)", 0, 1.0)
    cols = [2, 44, 78, 120]
    head = [T(lang, "Cohort", "队列"), T(lang, "Role", "作用"), T(lang, "Adaptation labels drawn from", "适配标签来源"), T(lang, "Evaluation set", "评价集")]
    rows = [[T(lang, "MOVER, USA", "MOVER，美国"), T(lang, "Development source", "开发数据源"), T(lang, "45,994 development operations", "45,994 台开发手术"), T(lang, "4,000 operations (test subset)", "4,000 台（测试子集）")],
            [T(lang, "Local specialty cohort, China\n(abdominal tumor surgery > 1 h)", "本院专病队列，中国\n（腹部肿瘤手术，> 1 h）"), T(lang, "Temporal validation", "时间外验证"), T(lang, "4,326 operations, 2020–2024", "4,326 台，2020–2024 年"), T(lang, "2,753 operations, 2025–2026", "2,753 台，2025–2026 年")],
            [T(lang, "VitalDB, Korea", "VitalDB，韩国"), T(lang, "Second site", "第二站点"), T(lang, "4,357 training operations", "4,357 台训练手术"), T(lang, "1,885 operations (test split)", "1,885 台（测试划分）")]]
    ax.plot([2, 163], [32.5, 32.5], color=INK, lw=0.8); ax.plot([2, 163], [28.3, 28.3], color=INK, lw=0.4); ax.plot([2, 163], [6.0, 6.0], color=INK, lw=0.8)
    for x, h in zip(cols, head): ax.text(x, 30.4, h, va="center")
    for i, r in enumerate(rows):
        y = 24.5 - i * 6.6
        for x, c in zip(cols, r): ax.text(x, y, c, va="center", linespacing=1.15)
    ax.text(2, 2.8, T(lang, "Inputs: T1, structured preoperative variables in all cohorts. T2, T1 plus preoperative text (local preanesthesia note; VitalDB diagnoses and tests; MOVER exploratory).",
                      "输入：T1 为各队列共有的术前结构化变量。T2 为 T1 加术前文本（本院麻醉前访视记录、VitalDB 诊断与检查，MOVER 仅作探索）。"), va="center", color=GREY)
    # ---- b) four approaches as aligned pipelines
    ax = fig.add_axes([0, 0, 1, 0.60]); ax.set_xlim(0, 165); ax.set_ylim(0, 64); ax.axis("off"); panel(ax, "b)", 0, 1.0)
    names = [(T(lang, "Generative zero-shot", "生成式零样本"), GREY), (T(lang, "Minimal decision model", "最小决策模型"), SKY),
             (T(lang, "Calibrated decision model", "校准决策模型"), BLUE), (T(lang, "Task LoRA", "任务 LoRA"), VERM)]
    model = [T(lang, "Qwen3.5-9B\nfrozen", "Qwen3.5-9B\n权重冻结")] * 3 + [T(lang, "Qwen3.5-9B\n+ LoRA adapter", "Qwen3.5-9B\n+ LoRA 适配器")]
    cost = [T(lang, "≈89 output tokens\n7,067 ms per case", "约 89 个输出词元\n每例 7,067 ms"), T(lang, "no output tokens\n111 ms per case", "不生成词元\n每例 111 ms"),
            T(lang, "3 cutpoints fitted on k labels\nseconds on a CPU", "用 k 个标签拟合 3 个切点\nCPU 上数秒"), T(lang, "adapter trained on n labels\n6.7 min at n = 1,000", "用 n 个标签训练适配器\nn = 1,000 时 6.7 min")]
    for i, ((nm, col), md, cs) in enumerate(zip(names, model, cost)):
        y0 = 50 - i * 14.5; yc = y0 + 4.5
        ax.plot([2, 2], [y0 + 0.8, y0 + 8.2], color=col, lw=2.2, solid_capstyle="butt")
        ax.text(4.5, yc, nm, va="center")
        box(ax, 39, y0 + 1, 23, 7); ax.text(50.5, yc, T(lang, "Prompt + T1/T2", "提示 + T1/T2") if i else T(lang, "Prompt + T1\n(asks for rationale)", "提示 + T1\n（要求写理由）"), ha="center", va="center", linespacing=1.15)
        arrow(ax, 62, yc, 65, yc); box(ax, 65, y0 + 1, 22, 7); ax.text(76, yc, md, ha="center", va="center", linespacing=1.15)
        arrow(ax, 87, yc, 92, yc)
        if i == 0:   # free text
            ax.text(93, yc + 1.6, T(lang, "“…history of hypertension…”", "“……高血压病史……”"), va="center", style="italic", color=GREY)
            ax.text(93, yc - 1.8, T(lang, "Final answer: C → class", "Final answer: C → 类别"), va="center", color=GREY)
        else:        # option probabilities, optional cutpoints
            for j, pv in enumerate([0.07, 0.55, 0.32, 0.06]):
                ax.add_patch(Rectangle((93 + j * 3.6, y0 + 1.6), 2.4, pv * 9.5, fc=col, ec="none"))
                ax.text(94.2 + j * 3.6, y0 + 0.7, "ABCD"[j], ha="center", va="top", fontsize=FS - 1)
            if i == 2:
                l0, l1 = 109, 127
                ax.plot([l0, l1], [yc, yc], color=INK, lw=0.5)
                for v in (1, 2, 3, 4): xv = l0 + (v - 1) / 3 * (l1 - l0); ax.plot([xv, xv], [yc - 0.7, yc + 0.7], color=INK, lw=0.5); ax.text(xv, yc - 1.6, str(v), ha="center", va="top", fontsize=FS - 1)
                for cp in (1.8, 2.2, 2.8): xv = l0 + (cp - 1) / 3 * (l1 - l0); ax.plot([xv, xv], [yc - 1.2, yc + 1.8], color=BLUE, lw=1.0)
                ax.text((l0 + l1) / 2, yc + 2.6, "E = Σ j·p$_j$", ha="center", va="bottom", fontsize=FS - 0.5)
                arrow(ax, 107.2, yc, 108.6, yc)
            else:
                ax.text(110, yc, "arg max", va="center")
        ax.text(132, yc, cs, va="center", linespacing=1.2, color=INK)
    ax.plot([131, 131], [3, 60], color=GREY, lw=0.4)
    ax.text(2, 0.8, T(lang, "All arms use one base model. Evaluation: quadratic-weighted κ against the recorded ASA class, exact accuracy, ASA IV–V recall, latency and labels needed.",
                      "各臂共用同一底模。评价：相对记录 ASA 分级的二次加权 κ、精确准确率、ASA IV–V 召回率、延迟与所需标签数。"), va="center", color=GREY)
    save(fig, "fig1_design", lang)


# ================================================================ Fig. 2 flow (CONSORT style)
def fig_flow(lang):
    style(lang); fig = plt.figure(figsize=(W2, 150 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 165); ax.set_ylim(0, 150); ax.axis("off")
    cols = [(1, T(lang, "Local specialty cohort", "本院专病队列"), T(lang, "abdominal tumor surgery > 1 h,\nno endoscopic or hysteroscopic surgery", "腹部肿瘤手术 > 1 h，\n不含内镜与宫腔镜手术")),
            (55, "MOVER", T(lang, "University of California, Irvine", "加州大学尔湾分校")),
            (109, "VitalDB", T(lang, "Seoul National University Hospital,\nnoncardiac surgery", "首尔大学医院，非心脏手术"))]
    w, h = 30, 11
    main = [[T(lang, "Anesthesia overview\n8,852 operations", "手术麻醉概览\n8,852 台"), T(lang, "Patient table\n64,354 operations", "患者信息表\n64,354 台"), T(lang, "Cases\n6,388 operations", "病例\n6,388 台")],
            [T(lang, "Start time and ASA\nrecorded\n7,348 (6,828 patients)", "有麻醉开始时间与 ASA\n7,348 台（6,828 人）"), T(lang, "ASA I–V, deduplicated\n57,504", "ASA I–V，去重\n57,504 台"), T(lang, "ASA I–V\n6,242 (5,961 patients)", "ASA I–V\n6,242 台（5,961 人）")],
            [T(lang, "Age ≥ 18 years\n7,332", "年龄 ≥ 18 岁\n7,332 台"), T(lang, "Age ≥ 18 years\n57,473\n(38,245 patients)", "年龄 ≥ 18 岁\n57,473 台\n（38,245 人）"), None],
            [T(lang, "Preanesthesia visit\nwithin 30 days\n7,080", "术前 30 天内\n有麻醉前访视\n7,080 台"), None, None],
            [T(lang, "Analysis set\n7,079 (6,804 patients)", "分析集\n7,079 台（6,804 人）"), T(lang, "Split by patient", "按患者划分"), T(lang, "Split by patient", "按患者划分")]]
    excl = {(0, 1): T(lang, "No start time or\nASA, n = 1,504", "缺开始时间\n或 ASA，1,504 台"), (1, 1): T(lang, "No ASA or\nother, n = 6,850", "缺 ASA 等\n6,850 台"),
            (2, 1): T(lang, "ASA missing\nor 6, n = 146", "ASA 缺失或\n6 级，146 台"), (0, 2): T(lang, "Age < 18 years\nn = 16", "< 18 岁\n16 台"),
            (1, 2): T(lang, "Age < 18 years\nn = 31", "< 18 岁\n31 台"), (0, 3): T(lang, "No visit within\n30 days, n = 252", "30 天内无访视\n252 台"),
            (0, 4): T(lang, "Cross-period\nre-operation, n = 1", "跨期再手术\n1 台")}
    final = [[T(lang, "Development\n2020–2024\n4,326", "开发集\n2020–2024 年\n4,326 台"), T(lang, "Temporal\nvalidation\n2025–2026\n2,753\n(2,640 patients)", "时间外验证\n2025–2026 年\n2,753 台\n（2,640 人）")],
             [T(lang, "Development\n80%\n45,994", "开发\n80%\n45,994 台"), T(lang, "Test 20%\n11,479; evaluation\nsubset 4,000\n(2,606 patients)", "测试 20%\n11,479 台，评价\n子集 4,000 台\n（2,606 人）")],
             [T(lang, "Training\n70%\n4,357", "训练\n70%\n4,357 台"), T(lang, "Test 30%\n1,885\n(1,788 patients)", "测试 30%\n1,885 台\n（1,788 人）")]]
    ys = [123, 106, 89, 72, 55]
    for ci, (x, title, sub) in enumerate(cols):
        ax.text(x, 148, title, fontweight="bold", va="top"); ax.text(x, 144.2, sub, va="top", color=GREY, linespacing=1.2)
        last = None
        for ri, row in enumerate(main):
            if row[ci] is None: continue
            y = ys[ri]; box(ax, x, y, w, h); ax.text(x + w / 2, y + h / 2, row[ci], ha="center", va="center", linespacing=1.2)
            if last is not None:
                arrow(ax, x + w / 2, ys[last], x + w / 2, y + h)
                if (ci, ri) in excl:
                    ym = (ys[last] + y + h) / 2; ax.plot([x + w / 2, x + w + 2], [ym, ym], color=INK, lw=0.5)
                    box(ax, x + w + 2, ym - 4.2, 20, 8.4, lw=0.4); ax.text(x + w + 12, ym, excl[(ci, ri)], ha="center", va="center", linespacing=1.15, fontsize=FS - 0.5)
            last = ri
        cx = x + w / 2; bw = 24; lx = x - 0.5; rx = lx + bw + 2
        ax.plot([cx, cx], [ys[4], 48], color=INK, lw=0.5); ax.plot([lx + bw / 2, rx + bw / 2], [48, 48], color=INK, lw=0.5)
        for bx, txt in zip((lx, rx), final[ci]):
            arrow(ax, bx + bw / 2, 48, bx + bw / 2, 43); box(ax, bx, 21, bw, 22, lw=0.8)
            ax.text(bx + bw / 2, 32, txt, ha="center", va="center", linespacing=1.2)
    ax.text(2, 14, T(lang, "No patient appears in more than one set within a data source. Numbers are operations unless stated otherwise.",
                     "同一数据源内没有患者同时出现在两个集合中。除注明外，数字均为手术台数。"), color=GREY)
    save(fig, "fig_flow", lang)


# ================================================================ Fig. 3 speed and label efficiency
def fig_efficiency(lang):
    style(lang); fig, ax = plt.subplots(1, 3, figsize=(W2, 58 * MM), gridspec_kw=dict(wspace=0.42))
    pts = [(T(lang, "Generative", "生成式"), 7067, 0.227, GREY, "o", (-4, 6), "right"), (T(lang, "Minimal", "最小决策模型"), 111, 0.443, SKY, "o", (6, 0), "left"),
           ("", 111, 0.526, BLUE, "s", (0, 0), "left"), (T(lang, "Calibrated, 100 and 1,000 labels", "校准，100 与 1,000 标签"), 111, 0.534, BLUE, "s", (6, -5), "left"),
           (T(lang, "LoRA, 1,000 labels", "LoRA，1,000 标签"), 179, 0.559, VERM, "^", (6, 6), "left")]
    for n, x, y, c, m, off, ha in pts:
        ax[0].plot(x, y, m, color=c, ms=4); ax[0].annotate(n, (x, y), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=FS - 0.5)
    ax[0].set_xscale("log"); ax[0].set_xlim(60, 15000); ax[0].set_ylim(0.15, 0.62)
    ax[0].set_xlabel(T(lang, "Median latency per case (ms)", "单例延迟中位数（ms）")); ax[0].set_ylabel(T(lang, "Quadratic-weighted κ", "二次加权 κ"))
    def curve(a, ks, km, klo, khi, ln, lm, llo, lhi, xn, xm, title):
        a.fill_between([1] + ks, klo, khi, color=BLUE, alpha=0.15, lw=0)
        a.plot([1] + ks, km, "-", color=BLUE); a.plot([1] + ks, km, "s", color=BLUE, ms=2.6)
        a.errorbar(ln, lm, yerr=[np.array(lm) - np.array(llo), np.array(lhi) - np.array(lm)], fmt="^-", color=VERM, ms=3.2, capsize=1.5, elinewidth=0.6)
        a.plot(xn, xm, "--", color=GREY, lw=0.8); a.plot(xn, xm, "o", color=GREY, ms=2.2)
        a.set_xscale("log"); a.set_ylim(0.2, 0.62); a.set_xlabel(T(lang, "Local labels", "本地标签数")); a.set_title(title, loc="left", pad=3)
        a.set_xticks([1, 10, 100, 1000, 10000]); a.set_xticklabels(["0", "10", "100", "1,000", "10,000"])
    curve(ax[1], [25, 50, 100, 200, 500, 1000], [0.443, 0.491, 0.516, 0.526, 0.528, 0.535, 0.540], [0.443, 0.384, 0.471, 0.474, 0.501, 0.520, 0.540], [0.443, 0.546, 0.547, 0.547, 0.544, 0.540, 0.540],
          [100, 300, 1000, 4326], [0.423, 0.542, 0.559, 0.561], [0.338, 0.530, 0.553, 0.556], [0.469, 0.556, 0.563, 0.570], [100, 300, 1000, 4326], [0.431, 0.467, 0.521, 0.549], T(lang, "Local validation set", "本院验证集"))
    curve(ax[2], [25, 50, 100, 200, 500, 1000], [0.473, 0.467, 0.495, 0.508, 0.517, 0.521, 0.524], [0.473, 0.372, 0.422, 0.476, 0.495, 0.516, 0.524], [0.473, 0.519, 0.530, 0.528, 0.529, 0.529, 0.524],
          [100, 300, 1000, 3000, 12000], [0.410, 0.411, 0.505, 0.563, 0.583], [0.394, 0.256, 0.462, 0.551, 0.580], [0.438, 0.500, 0.527, 0.575, 0.587], [100, 300, 1000, 3000, 12000], [0.234, 0.305, 0.395, 0.469, 0.504], T(lang, "MOVER test subset", "MOVER 测试子集"))
    ax[1].text(1.6, 0.575, T(lang, "Calibrated", "校准决策模型"), color=BLUE); ax[1].text(150, 0.33, "LoRA", color=VERM); ax[1].text(330, 0.43, "XGBoost", color=GREY)
    ax[2].text(1.6, 0.575, T(lang, "Calibrated", "校准决策模型"), color=BLUE); ax[2].text(2000, 0.60, "LoRA", color=VERM); ax[2].text(85, 0.236, "XGBoost", color=GREY, ha="right", va="center")
    ax[0].set_title(T(lang, "Speed and agreement, local", "速度与一致性，本院"), loc="left", pad=3)
    for a, s in zip(ax, ("a)", "b)", "c)")): panel(a, s, -0.22, 1.06)
    save(fig, "fig2_cost_efficiency", lang)


# ================================================================ Fig. 4 forest plot
def fig_forest(lang):
    style(lang); P = pd.read_csv(f"{R}/paired_differences.csv")
    def g(ds, hyp, c):
        r = P[(P.dataset == ds) & (P.hyp == hyp) & (P.contrast == c)]
        return None if r.empty else (float(r.iloc[0]["diff"]), float(r.iloc[0]["lo"]), float(r.iloc[0]["hi"]))
    DM, OJ = BLUE, AMBER
    spec = [("H1", "Minimal decision model, local", "最小决策模型，本院", "本院 val（生成式子集）", "H1", "读出 T1 未校准 − 生成式零样本", DM),
            ("H1", "OpenJev, calibrated, local", "OpenJev 校准，本院", "本院 val（生成式子集）", "H1", "OpenJev T1 校准 − 生成式零样本", OJ),
            ("H1", "Minimal decision model, MOVER", "最小决策模型，MOVER", "MOVER 子集（生成式子集）", "H1", "读出 T1 未校准 − 生成式零样本", DM),
            ("H1", "OpenJev, calibrated, MOVER", "OpenJev 校准，MOVER", "MOVER 子集（生成式子集）", "H1", "OpenJev T1 校准 − 生成式零样本", OJ),
            ("H2", "T1, local, three-seed mean (primary)", "T1，本院，三种子均值（主要）", "本院 val", "H2", "读出 T1 校准(1000) − LoRA@1000 三种子均值", DM),
            ("H2", "T1, local, default seed", "T1，本院，默认种子", "本院 val", "H2", "读出 T1 校准(1000) − LoRA@1000 种子20261017", DM),
            ("H2", "T1, local, seed 11 (label-matched)", "T1，本院，种子 11（标签匹配）", "本院 val", "H2", "读出 T1 校准(1000) − LoRA@1000 种子11", DM),
            ("H2", "OpenJev T1, local, three-seed mean", "OpenJev T1，本院，三种子均值", "本院 val", "H2", "OpenJev T1 校准(1000) − LoRA@1000 三种子均值", OJ),
            ("H2", "T2, local, single seed", "T2，本院，单种子", "本院 val", "H2", "读出 T2 校准(1000) − LoRA T2@1000（单种子）", DM),
            ("H2", "OpenJev T2, local, single seed", "OpenJev T2，本院，单种子", "本院 val", "H2", "OpenJev T2 校准(1000) − LoRA T2@1000（单种子）", OJ),
            ("H2", "T1, MOVER, three-seed mean", "T1，MOVER，三种子均值", "MOVER 子集", "H2", "读出 T1 校准(1000) − LoRA@1000 三种子均值", DM),
            ("H2", "OpenJev T1, MOVER, three-seed mean", "OpenJev T1，MOVER，三种子均值", "MOVER 子集", "H2", "OpenJev T1 校准(1000) − LoRA@1000 三种子均值", OJ),
            ("H2", "T2, VitalDB, three-seed mean", "T2，VitalDB，三种子均值", "VitalDB", "H2", "读出 T2 校准(1000) − 目标站点 LoRA T2@1000（种子均值）", DM),
            ("H3", "Full LoRA, T1, local", "全量 LoRA，T1，本院", "本院 val", "H3", "LoRA@全量 三种子均值 − 读出 T1 校准(1000)", DM),
            ("H3", "Full LoRA vs OpenJev, T1, local", "全量 LoRA 对 OpenJev，T1，本院", "本院 val", "H3", "LoRA@全量 三种子均值 − OpenJev T1 校准(1000)", OJ),
            ("H3", "Full LoRA, T2, local", "全量 LoRA，T2，本院", "本院 val", "H3", "LoRA T2@全量（单种子）− 读出 T2 校准(1000)", DM),
            ("H3", "Full LoRA vs OpenJev, T2, local", "全量 LoRA 对 OpenJev，T2，本院", "本院 val", "H3", "LoRA T2@全量（单种子）− OpenJev T2 校准(1000)", OJ),
            ("H3", "LoRA@3,000, T1, MOVER", "LoRA@3,000，T1，MOVER", "MOVER 子集", "H3", "LoRA@3000 三种子均值 − 读出 T1 校准(1000)", DM),
            ("H3", "LoRA@12,000, T1, MOVER", "LoRA@12,000，T1，MOVER", "MOVER 子集", "H3", "LoRA@12000 三种子均值 − 读出 T1 校准(1000)", DM),
            ("H3", "LoRA@3,000, T2, VitalDB", "LoRA@3,000，T2，VitalDB", "VitalDB", "H3", "目标站点 LoRA T2@3000（种子均值）− 读出 T2 校准(1000)", DM)]
    rows = [(grp, en if lang == "en" else zh, g(d, h, c), col) for grp, en, zh, d, h, c, col in spec]; rows = [r for r in rows if r[2] is not None]
    N = json.load(open(f"{R}/review_numbers_105.json")); k = [i for i, r in enumerate(rows) if r[0] == "H2"][2]
    rows.insert(k + 1, ("H2", T(lang, "T1, local, 50 label pools", "T1，本院，50 个标签池"), (N["comb_diff"], N["comb_lo"], N["comb_hi"]), DM))
    ttl = dict(H1=T(lang, "H1  decision model − generative zero-shot", "H1  决策模型 − 生成式零样本"),
               H2=T(lang, "H2  calibrated (1,000 labels) − LoRA@1,000; margin −0.05", "H2  校准（1,000 标签）− LoRA@1,000，界值 −0.05"),
               H3=T(lang, "H3  LoRA − calibrated (1,000 labels)", "H3  LoRA − 校准（1,000 标签）"))
    cnt = [sum(r[0] == kk for r in rows) for kk in ("H1", "H2", "H3")]
    fig, axs = plt.subplots(3, 1, figsize=(W2, (0.16 * len(rows) + 1.25)), gridspec_kw=dict(height_ratios=cnt, hspace=0.95))
    fz = lambda v: "0.000" if abs(v) < 0.0005 else f"{v:+.3f}".replace("-", "−")
    for ax, kk, s in zip(axs, ("H1", "H2", "H3"), ("a)", "b)", "c)")):
        rr = [r for r in rows if r[0] == kk]
        for i, (_, lab, (d, lo, hi), col) in enumerate(rr):
            y = len(rr) - 1 - i; ax.plot([lo, hi], [y, y], color=col, lw=0.9); ax.plot(d, y, "s", color=col, ms=3)
            ax.text(1.02, y, f"{fz(d)} ({fz(lo)} to {fz(hi)})", transform=ax.get_yaxis_transform(), va="center")
        ax.set_yticks(range(len(rr))); ax.set_yticklabels([r[1] for r in rr][::-1]); ax.tick_params(axis="y", length=0)
        ax.axvline(0, color=INK, lw=0.5)
        if kk == "H2": ax.axvline(-0.05, color=VERM, lw=0.6, ls=(0, (3, 2)))
        ax.spines["left"].set_visible(False); ax.set_ylim(-0.7, len(rr) - 0.3); ax.set_title(ttl[kk], loc="left", pad=3)
        ax.set_xlabel(T(lang, "Difference in quadratic-weighted κ (95% CI)", "二次加权 κ 差值（95% CI）")); panel(ax, s, -0.47, 1.0)
    save(fig, "fig3_forest", lang)


# ================================================================ Fig. 5 VitalDB (single column)
def fig_vitaldb(lang):
    style(lang); fig, ax = plt.subplots(figsize=(W1, 62 * MM))
    ks = [1, 25, 50, 100, 200, 500, 1000]
    dm = ([0.380, 0.509, 0.533, 0.549, 0.555, 0.563, 0.567], [0.380, 0.395, 0.473, 0.491, 0.515, 0.554, 0.567], [0.380, 0.564, 0.566, 0.569, 0.569, 0.568, 0.567])
    oj = ([0.370, 0.509, 0.531, 0.554, 0.562, 0.572, 0.582], [0.370, 0.397, 0.474, 0.515, 0.529, 0.550, 0.582], [0.370, 0.570, 0.573, 0.581, 0.583, 0.584, 0.582])
    for (m, lo, hi), col, mk in ((dm, BLUE, "s"), (oj, AMBER, "o")):
        ax.fill_between(ks, lo, hi, color=col, alpha=0.09, lw=0); ax.plot(ks, m, "-", color=col); ax.plot(ks, m, mk, color=col, ms=2.4)
    ln, lm, ll, lh = [100, 300, 1000, 3000], [0.282, 0.470, 0.567, 0.584], [0.061, 0.365, 0.532, 0.561], [0.444, 0.570, 0.602, 0.599]
    ax.errorbar(ln, lm, yerr=[np.array(lm) - np.array(ll), np.array(lh) - np.array(lm)], fmt="^-", color=VERM, ms=3.2, capsize=1.5, elinewidth=0.6)
    ax.set_xscale("log"); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["0", "10", "100", "1,000"]); ax.set_ylim(0, 0.65)
    ax.set_xlabel(T(lang, "Target-site labels", "目标站点标签数")); ax.set_ylabel(T(lang, "Quadratic-weighted κ", "二次加权 κ"))
    ax.text(1.3, 0.60, T(lang, "Calibrated decision model", "校准决策模型"), color=BLUE); ax.text(1.3, 0.335, "OpenJev", color=AMBER); ax.text(130, 0.17, "LoRA", color=VERM)
    save(fig, "fig4_vitaldb", lang)


# ================================================================ Fig. 6 natural-language guidance
def fig_guidance(lang):
    style(lang)
    orig = [1.3, 47.0, 48.8, 2.9]
    mix = {"t1": [[11.1, 59.1, 27.4, 2.4], [43.1, 35.7, 20.2, 1.1], [0.0, 85.2, 14.6, 0.2], [64.8, 24.2, 10.5, 0.6]],
           "t2": [[10.4, 20.1, 59.5, 10.0], [8.7, 13.0, 70.4, 7.9], [0.0, 45.3, 50.6, 4.1], [18.4, 35.2, 38.9, 7.4]]}
    kap = {"t1": [0.443, 0.312, 0.272, 0.187], "t2": [0.505, 0.435, 0.557, 0.528]}
    cal, lora = {"t1": 0.534, "t2": 0.545}, {"t1": 0.561, "t2": 0.606}
    nm = [T(lang, "Recorded ASA", "原始记录"), T(lang, "No guidance", "无指导"), T(lang, "ASA 2020 definitions", "ASA 2020 官方定义"), T(lang, "Local recording habit", "本地记录习惯"), T(lang, "Strict guideline", "严格遵循指南")]
    seq = ["#DCE9F5", "#9CC3E4", "#4A90C8", "#0B4F8A"]
    fig = plt.figure(figsize=(W2, 100 * MM)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.62, wspace=0.30)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for a, tv, ttl in ((axA, "t1", T(lang, "T1, predicted class mix", "T1，预测类别构成")), (axB, "t2", T(lang, "T2, predicted class mix", "T2，预测类别构成"))):
        dist = [orig] + mix[tv]; bot = np.zeros(5)
        for k in range(4):
            v = np.array([x[k] for x in dist]); a.barh(range(5), v, left=bot, color=seq[k], height=0.62, ec="white", lw=0.3, label=["I", "II", "III", "IV–V"][k]); bot += v
        a.set_yticks(range(5)); a.set_yticklabels(nm if tv == "t1" else [""] * 5); a.invert_yaxis(); a.set_xlim(0, 100); a.tick_params(axis="y", length=0)
        a.set_xlabel("%"); a.set_title(ttl, loc="left", pad=3); a.spines["left"].set_visible(False)
        for i, kv in enumerate(kap[tv]): a.text(101.5, i + 1, f"κ {kv:.3f}", va="center")
    axA.legend(ncol=4, loc="upper left", bbox_to_anchor=(0.0, -0.32), handlelength=1.2, columnspacing=1.0, title=None)
    w = 0.34
    for j, (tv, col) in enumerate((("t1", "#9CC3E4"), ("t2", "#0B4F8A"))):
        bars = axC.bar(np.arange(4) + (j - 0.5) * w, kap[tv], w, color=col, label=tv.upper())
        for b_, kv in zip(bars, kap[tv]): axC.text(b_.get_x() + b_.get_width() / 2, kv - 0.03, f"{kv:.3f}", ha="center", va="top", fontsize=FS - 0.5, color="white" if tv == "t2" else INK)
        axC.hlines(cal[tv], -0.5, 3.5, color=col, ls=(0, (4, 2)), lw=0.8); axC.hlines(lora[tv], -0.5, 3.5, color=col, ls=(0, (1, 1.5)), lw=1.0)
    axC.text(3.52, cal["t2"] - 0.012, T(lang, "calibrated, 1,000 labels", "校准，1,000 标签"), va="top", fontsize=FS - 0.5)
    axC.text(3.52, lora["t2"] + 0.004, T(lang, "full local LoRA", "全量本院 LoRA"), va="bottom", fontsize=FS - 0.5)
    axC.set_xticks(range(4)); axC.set_xticklabels(nm[1:]); axC.set_ylabel("κ"); axC.set_ylim(0, 0.7); axC.set_xlim(-0.55, 4.35)
    axC.set_title(T(lang, "Quadratic-weighted κ by guidance condition (light bars, T1; dark bars, T2)", "各指导条件的二次加权 κ（浅色为 T1，深色为 T2）"), loc="left", pad=3)
    panel(axA, "a)", -0.42, 1.05); panel(axB, "b)", -0.10, 1.05); panel(axC, "c)", -0.065, 1.05)
    save(fig, "fig5_guidance", lang)


if __name__ == "__main__":
    for lang in ("en", "zh"):
        fig_design(lang); fig_flow(lang); fig_efficiency(lang); fig_forest(lang); fig_vitaldb(lang); fig_guidance(lang)
    print("figures written to", OUT)
