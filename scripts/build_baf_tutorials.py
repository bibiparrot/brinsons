"""Write paired English and Chinese BAF notebooks from shared code cells."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]

CODE = [
    """from pathlib import Path
import sys
import polars as pl
from IPython.display import display

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'data/baf/synthetic_daily_positions.csv').exists()), None)
if ROOT is None:
    raise FileNotFoundError('Run scripts/generate_baf_data.py first')
sys.path.insert(0, str(ROOT / 'src'))

# Jupyter may retain an older brinsons import after the project is edited.
import importlib
import brinsons
import brinsons.baf
importlib.reload(brinsons.baf)
importlib.reload(brinsons)

from brinsons import (
    prepare_baf, calculate_baf, stock_picking_decisions, sizing_decisions,
    entry_timing_decisions, scaling_in_decisions, size_adjusting_decisions,
    scaling_out_decisions, exit_timing_decisions, decision_metrics, plot_baf,
    payoff_scale, inverse_payoff_scale, normalized_edge,
    baf_change_table, chart_baf_change_table,
)
grid = pl.read_csv(ROOT / 'data/baf/synthetic_daily_positions.csv', try_parse_dates=True)
print('Daily rows:', grid.height, '| portfolios:', grid['portfolio'].n_unique())
display(grid.head(5))""",
    """prepared = prepare_baf(
    grid,
    column_portfolio='portfolio', column_date='date', column_security='security',
    column_shares='shares', column_price='price',
    column_benchmark_level='benchmark_level',
    open_episode_policy='error',
)
print('Closed episodes:', prepared.episodes.height)
display(prepared.episodes.head(10))""",
    """separate = {
    'stock_picking': stock_picking_decisions(prepared),
    'sizing': sizing_decisions(prepared),
    'entry_timing': entry_timing_decisions(prepared, forward_days=21),
    'scaling_in': scaling_in_decisions(prepared, phase_fraction=0.9),
    'size_adjusting': size_adjusting_decisions(prepared, phase_fraction=0.9),
    'scaling_out': scaling_out_decisions(prepared, phase_fraction=0.9),
    'exit_timing': exit_timing_decisions(prepared, forward_days=21),
}
display(pl.DataFrame({'decision_type': list(separate),
                      'n_decisions': [frame.height for frame in separate.values()]}))
display(separate['stock_picking'].head(5))""",
    """result = calculate_baf(
    prepared,
    entry_forward_days=21, exit_forward_days=21,
    scaling_in_fraction=0.9, size_adjusting_fraction=0.9,
    scaling_out_fraction=0.9, minimum_decisions=5,
    payoff_scale_exponent=1.5,
)
assert result.decisions.height == sum(frame.height for frame in separate.values())
display(result.decisions.head(12))
display(result.metrics.select('portfolio', 'decision_type', 'n_decisions',
                              'hit_rate', 'payoff', 'scaled_payoff',
                              'mean_value_added'))""",
    """# The background is E(h,P)=h*P-(1-h), computed without manager data.
exponent = result.payoff_scale_exponent
chance_payoff = (1 - 0.25) / 0.25
chance_scaled = payoff_scale(chance_payoff, exponent=exponent)
print('Chance curve at hit rate 0.25:', round(chance_scaled, 3))
assert abs(normalized_edge(0.25, inverse_payoff_scale(chance_scaled, exponent=exponent))) < 1e-12

fig, axes = plot_baf(
    result,
    contour_levels=(-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 5, 50),
    grid_size=161,
    cmap='Purples',
)
display(fig)
import matplotlib.pyplot as plt
plt.close(fig)""",
    """# A completed-episode window; the full grid remains available for post-exit prices.
from datetime import date
windowed = calculate_baf(prepared, window_start=date(2025, 4, 1),
                         entry_forward_days=21, exit_forward_days=21,
                         minimum_decisions=1)
print('Episodes fully inside the window:', windowed.prepared.episodes.height)
display(windowed.metrics.select('portfolio', 'decision_type', 'n_decisions',
                                'hit_rate', 'scaled_payoff'))""",
    """# Exhibit-8-style changes across common managers in two completed-episode windows.
from datetime import date
early = calculate_baf(prepared, window_end=date(2025, 4, 1))
late = calculate_baf(prepared, window_start=date(2025, 4, 1))
changes = baf_change_table(early, late)
display(changes)
fig, ax = chart_baf_change_table(
    changes, title='Synthetic BAF peer changes: first vs second window'
)
display(fig)
plt.close(fig)""",
]

TEXT = {
    "en": [
        "# Behavioral Alpha Frontier (BAF) — research tutorial\n\nThis notebook uses **fictional** daily holdings to illustrate a transparent BAF inspired by the published EBAB/EBAF description. It is not Essentia's proprietary implementation. Stock picking and sizing follow published *descriptions*; five other decision methods are explicitly named proxies. No official 0–100 score or peer ranking is produced. See [formulas and limitations](../../docs/source/behavioral-alpha-frontier.md).",
        "## 1. Load the local grid\n\nEach row is one portfolio, security and trading date, including zero-share dates after exit. Prices and shares use a consistent adjusted basis. All source field names can be passed directly to `prepare_baf`; no input renaming is needed.",
        "## 2. Construct closed episodes\n\nA sale to zero ends one episode. A later purchase starts a new episode for the same security. This synthetic grid contains two closed episodes per portfolio-security pair.",
        "## 3. Run seven named decision functions\n\nEach method has its own calculation entrance. `forward_days=21` means 21 observed trading rows. The three phase fractions control the build/adjust/exit split. The [method guide](../../docs/source/behavioral-alpha-frontier.md) specifies every counterfactual.",
        "## 4. Calculate hit rate and payoff\n\nFor decisions $y_i$, hit rate is $\\#\\{y_i\\ge0\\}/n$ and payoff is $\\operatorname{mean}(y_i\\mid y_i\\ge0)/|\\operatorname{mean}(y_i\\mid y_i<0)|$. This package uses $S_k(P)=P^k/(1+P^k)$ with default $k=1.5$. The paper does not publish its exact scaling transform. Undefined payoff is null. `all_decisions` is a descriptive pooled point.",
        "## 5. Visualize data-independent indifference bands\n\nColor bands are contours of **normalized edge** $E(h,P)=hP-(1-h)$ on a fixed $(h,S_k(P))$ grid. They are calculated independently of the manager dots. The thick dashed $E=0$ curve passes through $(0.5,0.5)$; with $k=1.5$ it is curved and has height about $0.839$ at $h=0.25$. The colored boundaries and exponent are illustrative assumptions, not official Essentia curves. Positive normalized edge is above/right of the chance curve; equal normalized edge need not mean equal raw return across managers.",
        "## 6. Optional completed-episode window\n\nPrepare from full history and then select a window. The paper uses quarterly 36-month rolling analysis and additional coverage rules; this example applies a simpler completed-episode filter.",
        "## 7. Visual peer-change table\n\nInspired by the layout of Exhibit 8, this table compares two windows for the **same eligible managers**. Columns show the current percentage minus the previous percentage for the share adding value (positive mean decision value), with hit rate over 50%, and with payoff over 1. Values are displayed with `%`; they are not relative growth rates. Green indicates an increase and pink a decrease. These are synthetic results from this implementation, not the paper's reported percentages.",
        "## Interpretation\n\nDecision values are **not additive Brinson effects**. They do not reconcile to portfolio active return. Synthetic results must not be compared with published empirical percentages or an Essentia Behavioral Alpha Score. Study the [equations](../../docs/source/behavioral-alpha-frontier.md) before using other data.",
    ],
    "zh": [
        "# 行为阿尔法前沿（BAF）研究教程\n\n本教程用**虚构**的每日持仓，演示参考公开 EBAB/EBAF 论文独立重建的 BAF。七类决策的公式均在[方法说明](../../docs/source/behavioral-alpha-frontier.md)中列出。论文没有公开全部商业算法和综合评分映射，因此本项目不声称复现官方分数或同业排名。",
        "## 1. 读取本地数据\n\n每行对应一个账户、证券和交易日，清仓后仍有零持仓价格行。价格与股数须采用一致的复权口径。字段名可直接传给 `prepare_baf`，无需重命名输入列。",
        "## 2. 构造完整持仓段\n\n清仓至零结束一个 episode；同一证券再次买入会开启新 episode。模拟数据中每个账户、证券组合都有两段完整持仓。",
        "## 3. 分别调用七类决策函数\n\n每类方法都有独立的计算入口。`forward_days=21` 表示 21 个有数据的交易日；三个 phase fraction 控制建仓、调整与退出阶段划分。各反事实定义见[方法说明](../../docs/source/behavioral-alpha-frontier.md)。",
        "## 4. 计算命中率与盈亏比\n\n对决策值 $y_i$，命中率为 $\\#\\{y_i\\ge0\\}/n$，盈亏比为 $\\operatorname{mean}(y_i\\mid y_i\\ge0)/|\\operatorname{mean}(y_i\\mid y_i<0)|$。本项目采用缩放函数 $S_k(P)=P^k/(1+P^k)$，默认 $k=1.5$；论文未公开精确变换。无法定义盈亏比时返回空值。`all_decisions` 是描述性合并点。",
        "## 5. 绘制与数据点无关的无差异色带\n\n背景色带是固定 $(h,S_k(P))$ 网格上的**归一化增值** $E(h,P)=hP-(1-h)$ 等值线，与经理散点数据无关。粗虚线 $E=0$ 经过 $(0.5,0.5)$；当 $k=1.5$ 时呈弯曲状，在 $h=0.25$ 处高度约 $0.839$。色带层级和缩放参数都只是示意假设，并非 Essentia 官方曲线。正增值区域在虚线右上方；不同经理的归一化增值相同不代表原始收益相同。",
        "## 6. 可选的完整 episode 时间窗口\n\n先从完整历史构造 episode，再按窗口筛选。论文采用每季度滚动 36 个月及额外数据覆盖条件；此处仅展示较简单的完整持仓段筛选。",
        "## 7. 可视化同业变化表\n\n参考论文 Exhibit 8 的**排版结构**，对同一批具备有效指标的经理比较两个窗口。三列分别是：实现正平均决策增值的账户占比、命中率超过 50% 的账户占比、Payoff 大于 1 的账户占比。数值为**本期百分比减去上期百分比**，显示为 `%`，不是相对增长率。绿色代表上升，粉色代表下降。这里是虚构代理结果，不是论文公布的百分比或商业 Decision Alpha。",
        "## 解释边界\n\n决策值**不是可加总的 Brinson 效应**，也不与组合超额收益勾稽。虚构结果不可与论文实证百分比或 Essentia 行为阿尔法分数比较。用于其他数据前请阅读[公式说明](../../docs/source/behavioral-alpha-frontier.md)。",
    ],
}


def main() -> None:
    for language, blocks in TEXT.items():
        cells = [nbf.v4.new_markdown_cell(blocks[0])]
        for heading, source in zip(blocks[1:-1], CODE):
            cells.extend([nbf.v4.new_markdown_cell(heading), nbf.v4.new_code_cell(source)])
        cells.append(nbf.v4.new_markdown_cell(blocks[-1]))
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        })
        path = ROOT / "tutorials" / language / "behavioral_alpha_frontier.ipynb"
        nbf.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()
