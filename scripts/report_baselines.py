"""Turn reports/eval_baselines_summary.json into a markdown report.

Usage:  python scripts/report_baselines.py [in.json] [out.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports/eval_baselines_summary.json"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports/eval_baselines.md"

MODELS = ["decision_tree", "random_forest", "xgboost", "logit_l1"]
NICE = {"expert_tree": "expert tree", "decision_tree": "decision tree",
        "random_forest": "random forest", "xgboost": "XGBoost", "logit_l1": "L1-logistic"}
SCEN = ["random", "far_random", "true_nonmine", "spatial"]
FSETS = ["full", "no_density", "expert_matched"]

d = json.loads(SRC.read_text())
SP_SIDE = ROOT / "reports/eval_baselines_spatial_leakage.json"
sp_side = json.loads(SP_SIDE.read_text()) if SP_SIDE.exists() else {}
out: list[str] = []
w = out.append

w("# 专家树 vs 标准 ML baseline")
w("")
w(f"来源：`{SRC.relative_to(ROOT)}`（`scripts/eval_baselines.py`）。")
w("所有 arm 共用同一批正样本、同一 held-out 测试 id、同一四种负采样、同样的 5 km 泄漏缓冲和随机种子；")
w("唯一变化的是模型。训练集固定为「KB 正样本 − 测试 id」＋ 300 个有化探覆盖的随机背景点，")
w("即专家树自己拟合的那一套。")
w("")

def mean_auc(scen: str, model: str, fs: str = "expert_matched"):
    xs = []
    for _, v in d.items():
        r = v.get(scen) or {}
        src = r if model == "expert_tree" else (r.get("featuresets") or {}).get(fs, {})
        if src.get(model):
            xs.append(src[model]["auc"])
    return sum(xs) / len(xs) if xs else None


w("## 摘要（9 金属平均 AUC，baseline 只用专家树同款特征）")
w("")
w("| 场景 | " + " | ".join(NICE[m] for m in ["expert_tree"] + MODELS) + " | 说明 |")
w("|---" * (len(MODELS) + 3) + "|")
_note = {
    "random": "训练/测试正样本空间重叠严重，见 §0",
    "far_random": "paper 表 1 用的就是这一列，同样受 §0 影响",
    "true_nonmine": "正负样本都是真实矿点，采样密度可比",
    "spatial": "南训北测，唯一真正测外推的一档",
}
for scen in SCEN:
    vals = {m: mean_auc(scen, m) for m in ["expert_tree"] + MODELS}
    top = max(v for v in vals.values() if v is not None)
    cells = [f"**{v:.3f}**" if v == top else (f"{v:.3f}" if v else "—") for v in vals.values()]
    w(f"| {scen} | " + " | ".join(cells) + f" | {_note[scen]} |")
w("")

# ── leakage first: it conditions how every number below should be read ──
w("## 0. 先看这个：训练/测试正样本的空间重叠")
w("")
w("特征是 5/10/50 km 邻域统计。测试正样本若离某个训练正样本只有几百米，两者特征窗口几乎重合，")
w("随机划分下的 AUC 测的就不是泛化，而是近重复检索。")
w("")
w("| 金属 | 到最近训练正样本中位距离 | 最小 | p90 | ≤5 km | ≤10 km |")
w("|---|---|---|---|---|---|")
for metal, v in d.items():
    lk = (v.get("_leakage") or {}).get("random_split") or {}
    if not lk:
        continue
    w(f"| {metal} | {lk['median_km']} km | {lk['min_km']} km | {lk['p90_km']} km | "
      f"{lk['frac_within_5km']:.0%} | {lk['frac_within_10km']:.0%} |")
w("")
sp_rows = [(m, (v.get("_leakage") or {}).get("spatial_split") or sp_side.get(m) or {})
           for m, v in d.items()]
sp_rows = [(m, r) for m, r in sp_rows if r]
if sp_rows:
    w("空间划分（南训北测）下的同一诊断——这是唯一没有该问题的一档：")
    w("")
    w("| 金属 | 北部测试点到最近南部训练点中位距离 | ≤10 km |")
    w("|---|---|---|")
    for m, r in sp_rows:
        w(f"| {m} | {r['median_km']} km | {r['frac_within_10km']:.0%} |")
    w("")

# ── AUC tables ──
def table(scen: str, fs: str) -> None:
    w(f"### {scen} · baseline 特征集 = `{fs}`")
    w("")
    w("| 金属 | " + " | ".join(NICE[m] for m in ["expert_tree"] + MODELS) + " |")
    w("|---" * (len(MODELS) + 2) + "|")
    best_count = {m: 0 for m in ["expert_tree"] + MODELS}
    for metal, v in d.items():
        r = v.get(scen) or {}
        arm = (r.get("featuresets") or {}).get(fs, {})
        vals = {}
        et = r.get("expert_tree")
        if et:
            vals["expert_tree"] = et["auc"]
        for m in MODELS:
            if arm.get(m):
                vals[m] = arm[m]["auc"]
        if not vals:
            continue
        top = max(vals.values())
        best_count[max(vals, key=vals.get)] += 1
        cells = []
        for m in ["expert_tree"] + MODELS:
            if m not in vals:
                cells.append("—")
            elif vals[m] == top:
                cells.append(f"**{vals[m]:.3f}**")
            else:
                cells.append(f"{vals[m]:.3f}")
        w(f"| {metal} | " + " | ".join(cells) + " |")
    means = {}
    for m in ["expert_tree"] + MODELS:
        xs = []
        for metal, v in d.items():
            r = v.get(scen) or {}
            src = r if m == "expert_tree" else (r.get("featuresets") or {}).get(fs, {})
            if src.get(m):
                xs.append(src[m]["auc"])
        means[m] = sum(xs) / len(xs) if xs else None
    w("| **均值** | " + " | ".join(f"{means[m]:.3f}" if means[m] else "—"
                                   for m in ["expert_tree"] + MODELS) + " |")
    w("")
    w("最佳次数：" + "、".join(f"{NICE[m]} {c}" for m, c in best_count.items() if c) + "。")
    w("")


w("## 1. AUC 对比")
w("")
w("`expert_matched` 是唯一严格同口径的一档：baseline 只拿到 10 个专家看得见的那些特征。")
w("`full` 给 baseline 全部原始特征（约 1100 列），`no_density` 在此基础上去掉 `n_samples_*`（采样密度）。")
w("")
for scen in SCEN:
    w(f"## {SCEN.index(scen) + 1}.{scen}")
    w("")
    for fs in FSETS:
        table(scen, fs)

DST.write_text("\n".join(out))
print(f"wrote {DST}  ({len(out)} lines)")
