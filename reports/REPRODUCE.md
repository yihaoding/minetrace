# 复现说明

本文全部数字由五个脚本产生，共用同一套划分与随机种子。一条命令跑完：

```bash
cd /mmfs1/data/group/pmc050/yding/gad_reasoning
sbatch scripts/run_all_evals.sh          # 约 3.5 小时；勿在登录节点直接运行
```

## 1 环境

| 组件 | 版本 |
|---|---|
| Python | 3.10.18（conda 环境 `geochem`） |
| 解释器绝对路径 | `/mmfs1/data/group/pmc050/yding/miniconda3/envs/geochem/bin/python` |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| xgboost | 3.2.0 |
| shap | 0.49.1 |
| reportlab / pypdfium2 / fonttools | 项目本地 `.cache/pylibs`（仅供出图，与结果无关） |

`xgboost`、`shap`、`reportlab` 等装在项目内 `.cache/pylibs`，脚本自行 `sys.path.insert`，
不污染共享 conda 环境。重建方式：

```bash
export PIP_CACHE_DIR=$PWD/.cache/pip
$PY -m pip install --target $PWD/.cache/pylibs shap reportlab pypdfium2 fonttools
```

## 2 数据

| 用途 | 路径 |
|---|---|
| 化探（5 类样品） | `datasets/geochemical/states/state_geochemical/gswa_all_*.csv` |
| 已知矿点（每金属） | `datasets/geochemical/states/sites_enriched/enriched_<El>_sites.csv` |
| 他矿种矿点（true-nonmine 负样本） | `datasets/geochemical/states/sites/no_gold/no_<el>_sites.csv` |
| 地球物理栅格 | `datasets/selected_layers/rasters/` |
| 构造/地质矢量 | `datasets/selected_layers/vectors/` |

## 3 固定参数（`scripts/eval_comprehensive.py` 顶部）

| 参数 | 值 | 含义 |
|---|---|---|
| `SEED` | 42 | 主种子；派生 seed = 42+k，各场景固定 |
| `MAX_TEST` | 30 | 每金属 held-out 正样本数 |
| `N_NEG` | 200 | 每场景负样本数 |
| `N_TRAIN_NEG` | 300 | 训练背景点数（= 专家树 `n_bg`） |
| `BUFFER_KM` | 5.0 | 负样本距 test 正样本的最小距离 |
| `FAR_KM` | 50.0 | far-random 距任何已知矿点的最小距离 |
| `SPATIAL_BOUNDARY` | −25.0 | 纬度分界：南训北测 |
| `N_BOOT` | 10 | 归因稳定性的 bootstrap 次数 |
| `N_REP` | 5 / 3 | 缺失模态 / 选择性预测的随机子集重复数 |

派生种子对照：`train_neg=142`、`random=43`、`far_random=44`、`true_nonmine=45`、
`spatial 负样本=47`、`spatial agent=52`、`bootstrap b=1042+b`。

## 4 脚本与产出

| 脚本 | 产出 | 耗时 |
|---|---|---|
| `eval_comprehensive.py` | `eval_comprehensive_summary.json`、`eval_scores.json` | ~45 min |
| `eval_baselines.py` | `eval_baselines_summary.json` | ~55 min |
| `eval_interpretability.py` | `eval_interpretability_summary.json` | ~30 min |
| `eval_explanation_quality.py` | `eval_explanation_quality.json` | ~50 min |
| `eval_degradation.py` | `eval_degradation.json` | ~15 min |
| `eval_confidence.py` | `eval_confidence.json` | ~15 min |
| `report_baselines.py` | `eval_baselines.md` | 秒级 |
| `md2pdf.py` | 任意 md → PDF（中文，无需 pandoc/latex） | 秒级 |

单金属调试：所有评测脚本均支持 `--metals Cu`。

## 5 论文数字 → 出处对照

| 论文中的数字 | 来源文件 | JSON 路径 |
|---|---|---|
| 专家树 9 金属 × 4 场景 AUC | `eval_baselines_summary.json` | `<metal>.<scenario>.expert_tree.auc` |
| baseline 同口径 AUC | 同上 | `<metal>.<scenario>.featuresets.expert_matched.<model>.auc` |
| 训练/测试正样本空间距离 | 同上 | `<metal>._leakage.random_split` |
| spatial 划分距离 | `eval_baselines_spatial_leakage.json` | `<metal>` |
| 根重构误差 = 0、权重查找失败率 = 0 | `eval_interpretability_summary.json` | `results.<metal>.additivity` / `.z_sanity` |
| 专家级 Hit@1 = 0.904 | 同上 | `results.<metal>.ablation_consistency.hit_at_1` |
| 归因稳定性 0.631 | `eval_explanation_quality.json` | `<metal>.<model>.stability_top3_jaccard` |
| 组级忠实度 ρ / Hit@1 | 同上 | `<metal>.<model>.faithfulness_group` |
| 解释单元数 n80 | 同上 | `<metal>.<model>.parsimony_n80` |
| confidence 随缺失单调（ρ=−1.00） | `eval_degradation.json` | `<metal>.curve.<k>._expert_confidence` |
| 选择性预测 auc@25、ρ_err | `eval_confidence.json` | `<metal>.<model>` |

## 6 已知问题

1. **REE 配置有缺陷**：化探表中无 `REE` 元素列（只有 Ce/Dy/Er/Eu 等单元素），
   目标富集专家空转，仅 34 个特征、7 个活跃专家。REE 的数字须标注此限制或先做元素聚合。
2. **Mn 在 spatial 上 AUC = 0.269**（显著低于随机，方向学反），须解释或剔除。
3. `eval_degradation.py` 曾在 REE 上因 None 格式化崩溃，已修；REE 一行需补跑。
4. **agent 层幻觉率评测从未运行**：`demo/paper` 的 Table 2 为占位符，`demo/eval/prompts`
   与 `demo/eval/results` 为空目录。
5. 训练/测试正样本之间无空间隔离（详见 `reports/summary_table.md`）。random / far-random /
   true-nonmine 应表述为近矿（brownfield）排序能力，spatial 才是跨区域外推。

## 7 复现校验

跑完后用以下断言核对（数值应逐位一致，随机性已全部固定）：

```python
import json
b = json.load(open('reports/eval_baselines_summary.json'))
assert abs(b['Cu']['far_random']['expert_tree']['auc'] - 0.925) < 1e-3
assert abs(b['Cu']['far_random']['featuresets']['expert_matched']['random_forest']['auc'] - 0.995) < 1e-3
e = json.load(open('reports/eval_explanation_quality.json'))
assert abs(e['Cu']['expert_tree']['stability_top3_jaccard'] - 0.649) < 0.02   # bootstrap 有微小抖动
i = json.load(open('reports/eval_interpretability_summary.json'))['results']
assert all(v['additivity']['root_recon_err_max'] == 0.0 for v in i.values())
```
