# Jupyter tutorials / Jupyter 教程

Run Jupyter from the project root or from `tutorials/en/` or `tutorials/zh/`.
Each tutorial reads local files under `data/`; its English and Chinese versions
use the same calculations.

从项目根目录或 `tutorials/en/`、`tutorials/zh/` 启动 Jupyter。各教程读取
`data/` 下的本地数据；中英文版本使用相同的计算流程。

| Topic / 主题 | English | 中文 |
|---|---|---|
| Single-period Brinson / 单期归因 | [Single period](en/single_period_attribution.ipynb) | [单期归因](zh/single_period_attribution.ipynb) |
| Multiple-period linking / 多期链接 | [Multiple periods](en/multiple_period_attribution.ipynb) | [多期链接](zh/multiple_period_attribution.ipynb) |
| Dow 30 interactive / 道琼斯 30 交互分析 | [Dow 30](en/dow30_brinson_interactive.ipynb) | [道琼斯 30](zh/dow30_brinson_interactive.ipynb) |
| Multi-account / 多账户归因 | [Multi-account](en/multi_account_attribution.ipynb) | [多账户](zh/multi_account_attribution.ipynb) |
| Behavioral Alpha Frontier / 行为阿尔法前沿 | [BAF](en/behavioral_alpha_frontier.ipynb) | [行为阿尔法前沿](zh/behavioral_alpha_frontier.ipynb) |

The single-period tutorial calls single-period calculation and reporting tools.
The multiple-period tutorial calls multiple-period calculation, comparison,
linking, charts and waterfalls. The Dow 30 and multi-account tutorials cover
their dedicated workflows and include explicit method labels.
The BAF tutorial uses fictional daily holdings under `data/baf/` and shows
seven independently specified decision calculations, a frontier, and a
peer-change table.
