# TESTING.md — Vibe-Coding 测试指南

本文件指导 **AI coder + human reviewer** 在 `demo/` 下增量开发并验证代码。
适用对象:Claude Code / Cursor / 任何代理式编码工具,以及人类合作者。

> 单一目标:让每一次提交的代码都能 **被外部进程独立运行并产生可见输出**,
> 而不是只能编译通过。Vibe coding 最大的风险是「看起来对」却没跑过。

---

## 0. 环境前置(每次新会话必须确认)

```bash
# Conda env
conda activate geochem            # /group/pmc050/yding/miniconda3/envs/geochem
# 或显式用 conda run:
conda run -n geochem python -c "import numpy, pandas, scipy, sklearn"

# 工作目录
cd /group/pmc050/yding/gad_reasoning           # ← 项目根
# 不要 cd demo/ 后再运行;脚本里假设 ROOT = .../gad_reasoning
```

**Sanity check(60 秒内必须全过)**:

```bash
conda run -n geochem python -c "from core.catalog import DataCatalog; print('core OK')"
conda run -n geochem python -c "from domains.geochem.narrative import describe; print('narrative OK')"
ls datasets/geochemical/states/state_geochemical/gswa_all_sediment.csv
ls datasets/geochemical/states/sites/gswa_au_site.csv
```

若任何一条失败 → **停下问人**,不要继续写代码。

---

## 1. 已有可运行入口(现状基线)

| 脚本 | 命令 | 期望输出 |
|------|------|----------|
| `demo/scripts/demo_narrative.py` | `cd demo && python scripts/demo_narrative.py` | stdout 打印 5 个点的 narrative(Cu) |
| `demo/scripts/demo_au_narrative.py` | `python demo/scripts/demo_au_narrative.py` | stdout + `demo/sessions/au_10_points_narrative.md` |

**在动任何新代码前**,先把以上两条跑通,确认基线无回归。
跑完贴出最后 10 行 stdout 给 reviewer。

---

## 2. 计划中的新模块测试模板

`demo/agent/` 下尚未实现的模块,按下表逐个落地。
**每加一个模块 → 立刻加一个 `demo/tests/test_<module>.py` 并跑通**,再写下一个。

### 2.1 模块对应测试清单

| 模块 (`demo/agent/`) | 必须验证的最小行为 | 测试文件 |
|----------------------|--------------------|----------|
| `regions.py` | bbox / circle / named region 三种构造 + `contains(lon,lat)` | `tests/test_regions.py` |
| `region_scoring.py` | 30×30 粗网格 + 100×100 细网格,返回 shape 正确 | `tests/test_region_scoring.py` |
| `case_study_loader.py` | 读 JSON 不丢字段,缺失文件抛清晰异常 | `tests/test_case_study_loader.py` |
| `model_cache.py` | LRU 命中/淘汰 + 9 个 metal 都能 lazy 加载 | `tests/test_model_cache.py` |
| `tools.py`(12 工具) | 每个 tool 单独有一个 test;输入 schema 校验通过;返回含 `audit` 字段 | `tests/test_tools.py` |
| `system_prompt.py` | 字符串包含「numeric claim must be quotable」类约束 | inline assert 即可 |
| `runner.py` | mock OpenAI client → tool-loop 能正确路由 + audit 累积 | `tests/test_runner.py` |

### 2.2 测试文件骨架(复制即用)

```python
# demo/tests/test_<module>.py
"""Unit tests for demo.agent.<module>. NO LLM calls. NO network."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # .../gad_reasoning
sys.path.insert(0, str(ROOT))

import pytest
from demo.agent.<module> import <symbol>


def test_<behavior>_<expected>():
    result = <symbol>(...)
    assert result.<field> == <expected_value>
    assert "audit" in result.__dict__   # for tool returns
```

### 2.3 运行测试

```bash
# 单文件
conda run -n geochem python -m pytest demo/tests/test_regions.py -v

# demo 全量
conda run -n geochem python -m pytest demo/tests/ -v

# 带覆盖率(可选)
conda run -n geochem python -m pytest demo/tests/ --cov=demo/agent --cov-report=term-missing
```

`pyproject.toml` 当前 `testpaths = ["tests"]` 只包含项目根 `tests/`。
**demo 测试需显式传路径**(如上),或新增 `demo/pytest.ini` 把 `demo/tests` 加入。

---

## 3. Vibe-coding 工作流(每个 PR 都走这五步)

1. **读 README** — 先读 `demo/README.md` 的 "Current status" 表,确认要做的是下一个未打勾项,不要重复或越界。
2. **先写测试** — 即便是 30 行的工具函数,先把 `test_<thing>.py` 的入参/出参形状写出来,跑一次让它红。
3. **最小实现** — 让测试转绿,**不**顺手重构无关代码,**不**加未来才用的抽象。
4. **回归** — 跑 §1 的两个基线脚本,确认 narrative 输出未变化。
5. **更新 README 状态表** — 把对应行从 ⬜ 改成 ✅,在 PR description 里贴上 pytest 通过截图/文本。

---

## 4. 必须避免的「看起来对」

| 反模式 | 正确做法 |
|--------|----------|
| 只跑 `python -c "import demo.agent.tools"` 就声称通过 | 必须有 assert 行为的测试 |
| 用 `try/except: pass` 把异常吞掉 | 写测试断言异常类型与消息 |
| 给 `score_point` 返回 mock 数字 | 走真实 `ProspectivityModel`,只 mock LLM |
| 在 `tools.py` 里直接调 OpenAI | OpenAI 调用只能在 `runner.py`;tools 是纯函数 |
| 用相对 import `from .regions import ...` 然后被脚本运行报错 | 一律 `from demo.agent.regions import ...`,脚本里 `sys.path.insert(0, ROOT)` |
| 在 tool 返回里省略 `audit` 字段 | **强制**:每个 tool 返回都带 `audit: dict`(见 README §三设计原则) |

---

## 5. 数据/numeric faithfulness 检查(评估阶段强制)

agent 跑出的每一个数字(z-score、g-score、坐标、距离),
**必须在同一个 turn 的 tool return 里 verbatim 出现**。

人工/CI 检查脚本(待实现 `scripts/eval_agent_grounding.py`):

```
1. 取 agent 最终回复文本
2. 正则提取所有数字 token (含小数、负号、单位)
3. 取该 turn 所有 tool returns 的 JSON 序列化字符串
4. assert 每个数字 token 都在 tool returns 里出现
5. 不在的 → 标 numeric hallucination,写入 eval/results/numeric_hallucinations.csv
```

新增 tool 时,自查:这个 tool 返回的字段,**能否被 agent 当作可引用的数字源**?
能就保留;只是中间计算量 → 不要进返回 JSON。

---

## 6. 提交前 checklist

- [ ] §0 sanity check 全过
- [ ] 新增/修改文件对应的 `demo/tests/test_*.py` 已新增且 `pytest -v` 全绿
- [ ] §1 两个基线脚本仍能跑出非空 stdout
- [ ] 新增 tool 返回带 `audit` 字段
- [ ] `demo/README.md` 状态表已更新
- [ ] 没有提交 `__pycache__/`、`.ipynb_checkpoints/`、本地数据文件

---

## 7. 出问题先看哪里

| 症状 | 先检查 |
|------|--------|
| `ModuleNotFoundError: core` | 没 `cd gad_reasoning` 或 `sys.path` 没加 ROOT |
| `FileNotFoundError: gswa_*.csv` | 没用 `geochem` env / 数据集软链断了 |
| narrative 输出全是 NaN | `SedimentSource(center_lat=...)` 没传(见 `project_gad.md` Stage 2 变更 #1) |
| Au AUC 显示 0.5x | 看 strategy:strategy-A 在 Au 上本身就低,见 README & memory,**不是 bug** |
| tool return 缺 audit | 你写错了,回到 §4 表 |

---

## 8. 跟人类 reviewer 的最小契约

提交 PR 时,description 里贴出:

```
### 改动
- 新增 demo/agent/<module>.py
- 新增 demo/tests/test_<module>.py

### 验证
$ conda run -n geochem python -m pytest demo/tests/test_<module>.py -v
<粘贴最后 20 行>

$ python demo/scripts/demo_au_narrative.py | tail -5
<粘贴最后 5 行,证明基线未坏>

### README 状态
demo/README.md 中 "<对应行>" 已从 ⬜ 改为 ✅
```

没有这三段输出 → reviewer 会直接打回,不要省。
