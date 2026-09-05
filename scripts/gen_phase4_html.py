"""Generate reports/phase4_summary.html — self-contained with embedded images."""
import base64, os

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(BASE, "reports")

with open(os.path.join(REPORTS, "phase4_roc.png"), "rb") as f:
    roc_b64 = base64.b64encode(f.read()).decode()
with open(os.path.join(REPORTS, "phase4_weights.png"), "rb") as f:
    wts_b64 = base64.b64encode(f.read()).decode()

CSS = """
  :root {
    --blue:    #3b82f6;
    --orange:  #f97316;
    --green:   #22c55e;
    --purple:  #a855f7;
    --red:     #ef4444;
    --light:   #f8fafc;
    --border:  #e2e8f0;
    --text:    #1e293b;
    --muted:   #64748b;
    --sa:      #2196F3;
    --sb:      #FF5722;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f1f5f9; color: var(--text); line-height: 1.6;
  }
  .page { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: .25rem; }
  .subtitle { color: var(--muted); font-size: .95rem; margin-bottom: 2rem; }
  h2 { font-size: 1.25rem; font-weight: 700; margin: 2.5rem 0 1rem;
       padding-bottom: .4rem; border-bottom: 2px solid var(--border); }
  h3 { font-size: .9rem; font-weight: 600; color: var(--muted);
       text-transform: uppercase; letter-spacing: .05em; margin-bottom: .5rem; }
  .card { background:#fff; border:1px solid var(--border); border-radius:10px;
          padding:1.4rem 1.6rem; margin-bottom:1.2rem; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  .grid4 { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1rem;
           margin-bottom:1.5rem; }
  @media(max-width:640px){ .grid2,.grid4 { grid-template-columns:1fr; } }
  .stat-card { background:#fff; border:1px solid var(--border); border-radius:10px;
               padding:1.1rem 1.4rem; }
  .stat-label { font-size:.78rem; color:var(--muted); text-transform:uppercase;
                letter-spacing:.06em; }
  .stat-value { font-size:1.9rem; font-weight:700; margin:.1rem 0; }
  .stat-sub   { font-size:.82rem; color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-size:.88rem; }
  th { background:var(--light); text-align:left; padding:.55rem .8rem;
       font-weight:600; border-bottom:2px solid var(--border); white-space:nowrap; }
  td { padding:.5rem .8rem; border-bottom:1px solid var(--border); vertical-align:top; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:#f8fafc; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  .badge { display:inline-block; border-radius:4px; padding:.15rem .5rem;
           font-size:.75rem; font-weight:600; white-space:nowrap; }
  .bg  { background:#dcfce7; color:#166534; }
  .bw  { background:#fee2e2; color:#991b1b; }
  .bo  { background:#fef9c3; color:#854d0e; }
  .bsa { background:#dbeafe; color:#1e40af; }
  .bsb { background:#ffedd5; color:#9a3412; }
  .t-Cu { border-left:4px solid var(--blue);   }
  .t-Au { border-left:4px solid var(--orange); }
  .t-W  { border-left:4px solid var(--green);  }
  .t-Ni { border-left:4px solid var(--purple); }
  .dot  { display:inline-block; width:10px; height:10px; border-radius:50%;
          margin-right:5px; vertical-align:middle; }
  .dot-Cu { background:var(--blue);   }
  .dot-Au { background:var(--orange); }
  .dot-W  { background:var(--green);  }
  .dot-Ni { background:var(--purple); }
  .bar-wrap { background:#e2e8f0; border-radius:4px; height:7px; width:80px;
              overflow:hidden; display:inline-block; vertical-align:middle; margin:0 6px; }
  .bar { height:100%; border-radius:4px; }
  .up   { color:#16a34a; font-weight:600; }
  .dn   { color:#dc2626; font-weight:600; }
  .flat { color:var(--muted); }
  img.fig { width:100%; border-radius:8px; border:1px solid var(--border); }
  .fc { background:#fff; border:1px solid var(--border); border-radius:8px;
        padding:1rem 1.2rem; }
  .fc.warn { border-left:4px solid var(--red);    }
  .fc.good { border-left:4px solid var(--green);  }
  .fc.info { border-left:4px solid var(--blue);   }
  .fc-title { font-weight:700; margin-bottom:.4rem; font-size:.95rem; }
  .fc-body  { font-size:.88rem; color:var(--muted); line-height:1.55; }
  .note { font-size:.85rem; color:var(--muted); font-style:italic; margin-bottom:1rem; }
  code { background:#f1f5f9; border-radius:3px; padding:.1rem .35rem;
         font-size:.85em; font-family:'SF Mono','Fira Code',monospace; }
  .footer { font-size:.8rem; color:var(--muted); text-align:center; margin-top:2rem; }
"""

def auc_bar(val, color):
    pct = max(0, min(100, (val - 0.5) * 200))
    return (f'<span class="bar-wrap"><span class="bar" '
            f'style="width:{pct:.0f}%;background:{color}"></span></span>')

def delta(val, ref):
    d = val - ref
    cls = "up" if d > 0.005 else ("dn" if d < -0.005 else "flat")
    sign = "+" if d >= 0 else ""
    return f'<span class="{cls}">{sign}{d:.3f}</span>'

rows = [
    # metal, ref,   a,     b
    ("Cu", 0.762, 0.764, 0.723),
    ("Au", 0.652, 0.560, 0.729),
    ("W",  0.718, 0.782, 0.749),
    ("Ni", 0.669, 0.798, 0.838),
]

judgement = {
    "Cu": ('<span class="badge bg">持平/略升</span>', "加回 rockchips 后与 Phase 3 持平"),
    "Au": ('<span class="badge bo">策略A偏低</span>', "Au 直接浓度异常对随机背景区分力弱"),
    "W":  ('<span class="badge bg">大幅提升</span>',  "rockchips 恢复后超越 Phase 3"),
    "Ni": ('<span class="badge bg">大幅提升</span>',  "地质+磁力+rockchips 三重加持"),
}

table_rows = ""
for metal, ref, a, b in rows:
    badge, note = judgement[metal]
    table_rows += f"""
        <tr class="t-{metal}">
          <td><span class="dot dot-{metal}"></span><strong>{metal}</strong></td>
          <td class="num">{ref:.3f}</td>
          <td class="num">{a:.3f} {auc_bar(a,'var(--sa)')} {delta(a,ref)}</td>
          <td class="num">{b:.3f} {auc_bar(b,'var(--sb)')} {delta(b,ref)}</td>
          <td>{badge}<br><small>{note}</small></td>
        </tr>"""

compare_rows = """
        <tr>
          <td><strong>Phase 2v2</strong></td>
          <td>沉积物（对比特征）</td>
          <td class="num">~0.70</td><td class="num">—</td>
          <td class="num">—</td><td class="num">—</td>
          <td>单目标基线</td>
        </tr>
        <tr>
          <td><strong>Phase 3</strong></td>
          <td>沉积物 + 岩屑</td>
          <td class="num">0.762</td><td class="num">0.652</td>
          <td class="num">0.718</td><td class="num">0.669</td>
          <td>多源地球化学</td>
        </tr>
        <tr style="color:#999;font-size:.82rem">
          <td><em>Phase 4 初版（无 rockchips）</em></td>
          <td><em>地化(沉积物) + 地物理 + 构造</em></td>
          <td class="num"><em>0.735</em></td>
          <td class="num" style="color:var(--red)"><em>0.548</em></td>
          <td class="num" style="color:var(--red)"><em>0.588</em></td>
          <td class="num"><em>0.794</em></td>
          <td><em>rockchips 遗漏版本，已废弃</em></td>
        </tr>
        <tr>
          <td><strong>Phase 4-A</strong></td>
          <td>地化(沉积物+岩屑) + 地物理 + 构造</td>
          <td class="num" style="color:var(--green)"><strong>0.764</strong></td>
          <td class="num">0.560</td>
          <td class="num" style="color:var(--green)"><strong>0.782</strong></td>
          <td class="num" style="color:var(--green)"><strong>0.798</strong></td>
          <td>随机负样本</td>
        </tr>
        <tr>
          <td><strong>Phase 4-B</strong></td>
          <td>地化(沉积物+岩屑) + 地物理 + 构造</td>
          <td class="num">0.723</td>
          <td class="num">0.729</td>
          <td class="num">0.749</td>
          <td class="num" style="color:var(--green)"><strong>0.838</strong></td>
          <td>跨金属负样本</td>
        </tr>"""

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 4 — Expert Tree Evaluation</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">

  <h1>Phase 4 — Expert Tree Evaluation</h1>
  <p class="subtitle">完整专家树（地球化学 + 地球物理 + 构造）× 4 金属 × 2 负样本策略 &nbsp;·&nbsp; 2026-05-07</p>

  <!-- Key metrics -->
  <div class="grid4">
    <div class="stat-card">
      <div class="stat-label">最佳 AUC</div>
      <div class="stat-value" style="color:var(--purple)">0.838</div>
      <div class="stat-sub">Ni — Strategy B（跨金属）</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">专家层数</div>
      <div class="stat-value">3</div>
      <div class="stat-sub">地球化学 · 地球物理 · 构造</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">叶级专家数</div>
      <div class="stat-value">11</div>
      <div class="stat-sub">地化 4（沉积物+岩屑各 2）+ 地物理 4 + 构造 3</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">超越 Phase 3</div>
      <div class="stat-value" style="color:var(--green)">3/4</div>
      <div class="stat-sub">Cu ≈ · W ↑↑ · Ni ↑↑（Au 策略A仍偏低）</div>
    </div>
  </div>

  <!-- Methodology -->
  <h2>方法论</h2>
  <div class="card">
    <div class="grid2">
      <div>
        <h3>专家树结构</h3>
        <p style="font-size:.9rem">每个金属一棵三层 <code>CompositeNode</code> 树：</p>
        <ul style="font-size:.88rem;margin-top:.5rem;padding-left:1.2rem;line-height:1.9">
          <li><strong>地球化学</strong>：<code>TargetEnrichmentExpert</code> + <code>TargetPathfinderExpert</code>（沉积物）</li>
          <li><strong>地球物理</strong>：磁力 · 重力 · 放射性 · 地质年代（4 栅格）</li>
          <li><strong>构造</strong>：断层 · 蠕虫线 · 地质背景</li>
        </ul>
        <p style="font-size:.88rem;margin-top:.6rem">权重：shape weight = <code>(AUC − 0.5) × 2</code></p>
      </div>
      <div>
        <h3>评估设计</h3>
        <ul style="font-size:.88rem;padding-left:1.2rem;line-height:1.9">
          <li><strong>验证集</strong>：每金属取 5%（≥30）已知矿点</li>
          <li><strong>训练集</strong>：剩余 95%，用跨金属站点作背景</li>
          <li><strong>策略 A</strong>：随机 WA 坐标（距已知站点 &gt;20 km）</li>
          <li><strong>策略 B</strong>：跨金属站点（其他金属矿点作负样本）</li>
          <li><strong>指标</strong>：Validation AUC（<code>roc_auc_score</code>）</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- AUC Table -->
  <h2>AUC 汇总</h2>
  <p class="note">括号内为 Phase 3（纯地球化学，沉积物 + 岩屑）参考 AUC；↑↓ 为相对变化。</p>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>金属</th>
          <th class="num">Phase 3 参考</th>
          <th class="num">Phase 4 <span class="badge bsa">策略 A 随机</span></th>
          <th class="num">Phase 4 <span class="badge bsb">策略 B 跨金属</span></th>
          <th>判断</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <!-- ROC Curves -->
  <h2>ROC 曲线</h2>
  <p class="note">上行：策略 A（随机负样本）；下行：策略 B（跨金属负样本）。列顺序：Au · Cu · Ni · W。</p>
  <div class="card" style="padding:1rem">
    <img class="fig" src="data:image/png;base64,{roc_b64}" alt="Phase 4 ROC Curves">
  </div>

  <!-- Weights -->
  <h2>叶级专家权重分布</h2>
  <p class="note">Shape weight = (训练 AUC − 0.5) × 2；接近 0 意味着该专家对当前金属判别无贡献。</p>
  <div class="card" style="padding:1rem">
    <img class="fig" src="data:image/png;base64,{wts_b64}" alt="Phase 4 Expert Weights">
  </div>

  <!-- Weight Interpretation -->
  <h2>权重解读</h2>
  <div class="grid4">
    <div class="stat-card t-Au" style="padding:1rem 1.2rem">
      <div class="stat-label" style="color:var(--orange)">Au</div>
      <p style="font-size:.88rem;margin-top:.4rem;line-height:1.6">
        <strong>au_sediment_pathfinder</strong> 仍主导（~0.32），<strong>au_rockchips_pathfinder</strong> 新增贡献。
        <strong>au_rockchips_enrichment</strong> 权重几乎为零——Au 在岩屑中的直接浓度异常同样不能很好区分随机背景，
        这与 Au 矿化的分散性（ppb 量级）和 WA 背景复杂性有关。
      </p>
    </div>
    <div class="stat-card t-Cu" style="padding:1rem 1.2rem">
      <div class="stat-label" style="color:var(--blue)">Cu</div>
      <p style="font-size:.88rem;margin-top:.4rem;line-height:1.6">
        <strong>radiometric_geophysics</strong> 最高，<strong>cu_rockchips_enrichment</strong> 上升至第二位。
        <strong>geology_structure / fault_structure / cu_sediment_pathfinder</strong> 接近。
        rockchips 的加入使 Cu AUC 完全追平 Phase 3（0.764 vs 0.762）。
      </p>
    </div>
    <div class="stat-card t-Ni" style="padding:1rem 1.2rem">
      <div class="stat-label" style="color:var(--purple)">Ni</div>
      <p style="font-size:.88rem;margin-top:.4rem;line-height:1.6">
        <strong>geology_structure</strong> 独占最高权重（~0.50），符合超镁铁岩控矿逻辑。
        <strong>ni_rockchips_enrichment / pathfinder</strong> 现在也有实质贡献，
        加上 <strong>magnetic_geophysics</strong>（~0.28），形成地质+地球物理+地化三层协同。
      </p>
    </div>
    <div class="stat-card t-W" style="padding:1rem 1.2rem">
      <div class="stat-label" style="color:var(--green)">W</div>
      <p style="font-size:.88rem;margin-top:.4rem;line-height:1.6">
        <strong>w_rockchips_enrichment</strong> 跃升为第一位（~0.62）。
        <strong>w_sediment_pathfinder</strong>（~0.55）和 <strong>radiometric_geophysics</strong>（~0.38）次之。
        W 矽卡岩矿化局部性强，沉积物稀释效应明显，岩屑才是核心信号来源。
      </p>
    </div>
  </div>

  <!-- Key Findings -->
  <h2>关键发现</h2>
  <div class="grid2">
    <div class="fc good">
      <div class="fc-title">W — 大幅恢复（0.588 → 0.782）</div>
      <div class="fc-body">
        加回 rockchips 后 W 直接超越 Phase 3（0.718）。权重图确认 <strong>w_rockchips_enrichment</strong>
        跃升为第一位（~0.62），印证了 W 矿化信号主要集中在岩石直接采样中，
        而非分散的沉积物数据——W 的矿化往往局部性强，河流沉积稀释效应明显。
      </div>
    </div>
    <div class="fc good">
      <div class="fc-title">Ni — 持续提升（0.669 → 0.838）</div>
      <div class="fc-body">
        <strong>geology_structure</strong> 专家权重最高（~0.50），符合超镁铁岩控矿逻辑。
        rockchips 的加入使 <strong>ni_rockchips_enrichment / pathfinder</strong> 现在也贡献有效信号，
        与地球物理层形成多层协同。
      </div>
    </div>
    <div class="fc warn">
      <div class="fc-title">Au — 策略 A 仍偏低（AUC 0.560）</div>
      <div class="fc-body">
        加回 rockchips 后 Au 改善有限（0.548 → 0.560）。权重图显示 <strong>au_rockchips_enrichment</strong>
        权重几乎为零——Au 的岩屑信号在 WA 背景中同样不突出，这与 Au 矿化的分散性和金的低背景浓度有关。
        策略 B（0.729）正常，说明问题是"vs 随机背景"的特异性不足，不是模型本身的问题。
      </div>
    </div>
    <div class="fc info">
      <div class="fc-title">rockchips 遗漏是 Phase 4 初版最大的设计缺陷</div>
      <div class="fc-body">
        Phase 4 代码从头重写时，<code>build_tree</code> 地球化学层只填了 <code>"sediment"</code>，
        rockchips 被漏掉。<code>SedimentSource</code> 类本来就支持任意 assay CSV，
        只需加一个 <code>name="rockchips"</code> 实例即可。W 的 AUC 从 0.588 跳至 0.782 直接说明了这一遗漏的代价。
      </div>
    </div>
  </div>

  <!-- Phase Comparison -->
  <h2>阶段进展对比</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>阶段</th><th>数据层</th>
          <th class="num">Cu</th><th class="num">Au</th>
          <th class="num">W</th><th class="num">Ni</th><th>备注</th>
        </tr>
      </thead>
      <tbody>{compare_rows}</tbody>
    </table>
  </div>

  <!-- Next Steps -->
  <h2>待办 / 下一步</h2>
  <div class="card">
    <table>
      <thead>
        <tr><th>#</th><th>任务</th><th>优先级</th><th>目的</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>Au + W 专家 ablation：仅地化 vs 地化+地物理 vs 完整树</td>
          <td><span class="badge bw">高</span></td>
          <td>定位哪层专家拖低了 AUC</td>
        </tr>
        <tr>
          <td>2</td>
          <td>W 各地球物理/构造专家单独训练 AUC 分析</td>
          <td><span class="badge bw">高</span></td>
          <td>确认是单个专家还是融合策略的问题</td>
        </tr>
        <tr>
          <td>3</td>
          <td>引入权重截断（weight threshold，如 &lt;0.05 的专家不参与融合）</td>
          <td><span class="badge bo">中</span></td>
          <td>防止 AUC ≈ 0.5 的专家稀释树输出</td>
        </tr>
        <tr>
          <td>4</td>
          <td>加入 rockchips 数据作为地球化学层输入（Phase 3 贡献显著）</td>
          <td><span class="badge bo">中</span></td>
          <td>当前 Phase 4 地化层仅用沉积物，低于 Phase 3 配置</td>
        </tr>
        <tr>
          <td>5</td>
          <td>解决 DP-01（Zhou 2014 Example 3.6 期望值）</td>
          <td><span class="badge bg">低</span></td>
          <td>BK 融合测试完整性，不影响主流程</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p class="footer">GAD Reasoning Project &nbsp;·&nbsp; Phase 4 Report &nbsp;·&nbsp; 2026-05-07</p>
</div>
</body>
</html>"""

out = os.path.join(REPORTS, "phase4_summary.html")
with open(out, "w") as f:
    f.write(html)
print(f"Written: {out}  ({len(html):,} bytes)")
