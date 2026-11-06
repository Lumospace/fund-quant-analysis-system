# 基金投资量化分析系统 · FundQuant

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://fund-quant-analysis-system-cz2tevzrrurdkumogkzxs7.streamlit.app/)

面向个人投资者的公募基金量化分析工作台。基于 AKShare 公开行情数据，覆盖**基金画像 → 筛选优选 → 横向对比 → 组合优化 → 大类资产配置 → 定投/回测 → 压力测试 → 一键报告**的完整投研链路，界面为 Streamlit 多页应用。

**在线演示**：<https://fund-quant-analysis-system-cz2tevzrrurdkumogkzxs7.streamlit.app/>
（托管于 Streamlit Community Cloud 免费版，应用休眠后首次访问需等待约 30 秒唤醒）

> 所有指标、模型参数、评分权重均在 [config.py](config.py) 中显式可查、可调，不做黑箱结论。
> 本项目仅提供量化分析工具，**不构成任何投资建议**。

---

## 目录

- [功能总览](#一功能总览)
- [量化方法](#二量化方法)
- [快速开始](#三快速开始)
- [项目结构](#四项目结构)
- [关键配置](#五关键配置)
- [测试](#六测试)
- [自行部署（可选）](#七自行部署可选)
- [数据来源与免责声明](#八数据来源与免责声明)
- [许可证](#九许可证)

---

## 一、功能总览

| 模块 | 页面路径 | 核心能力 |
|---|---|---|
| 数据驾驶舱 | `/dashboard` | 场内/场外基金池浏览、实时行情、行业板块分布、涨跌结构 |
| 基金画像 | `/profile` | 8 项核心指标、四维评分与评级、PE 估值分位、持仓穿透、因子诊断、AI 多周期预测 |
| 基金筛选 | `/screener` | 按类型/规模/收益/回撤/评分多维条件筛出候选基金 |
| 定投测算 | `/dca` | 固定定投 vs 估值定投，成本摊薄与收益对比 |
| 策略回测对比 | `/backtest` | 买入持有 / 双均线 / 动量 / 定投 策略横向回测 |
| 压力测试（单基金） | `/stress_single` | 历史极端情景回放 + 蒙特卡洛模拟，输出回撤、缩水幅度、修复周期 |
| 一键量化报告 | `/report` | 单基金完整 HTML 报告导出（指标、评分、预测、压力测试、操作建议） |
| 基金对比 | `/compare` | 多基金指标横向对比、四维评分雷达、最优标的结论，支持报告导出 |
| 组合构建 | `/portfolio` | 最大夏普 / 最小方差 / 目标收益 / 风险平价四类优化，风险贡献分解与样本外滚动回测 |
| 大类资产配置 | `/allocation` | 股 / 债 / 金 / 现四类资产配置与再平衡回测，支持报告导出 |
| 压力测试（组合） | `/stress_portfolio` | 组合层面情景压力测试与分散化收益测算，支持报告导出 |

---

## 二、量化方法

- **收益与风险指标**：年化收益（按日历日折算）、年化波动率、最大回撤、回撤修复周期、夏普比率、索提诺比率、卡玛比率。
- **估值分位**：优先使用对应宽基指数的 PE-TTM 历史（乐咕乐股），按研究区间末取分位；无对应指数时降级为价格分位。
- **四维评分**：收益 35% / 风险 35% / 估值 20% / 稳定性 10%，每维度给出可读的打分依据。
- **AI 预测**：LightGBM 多周期（短/中/长）概率预测，输出上涨/下跌概率、置信度与收益区间；超参数随样本量自适应。
- **因子诊断**：因子 IC 时序、累计 IC、分层回测、因子相关性矩阵与共线性预警。
- **组合优化**：`scipy` 求解权重，附风险贡献分解与滚动样本外回测，避免只看样本内最优化结果。
- **压力测试**：历史情景（含地缘/流动性冲击）+ 蒙特卡洛兜底，情景参数集中在 `config.STRESS_SCENARIOS`。

---

## 三、快速开始

环境要求：Python 3.10 及以上，可访问公网（用于拉取行情数据）。

```bash
git clone https://github.com/Lumospace/fund-quant-analysis-system.git
cd fund-quant-analysis-system

pip install -r requirements.txt
streamlit run app.py
```

浏览器打开终端输出的地址（默认 `http://localhost:8501`）。

首次运行需要联网拉取公开行情数据，结果缓存在 `.cache/`（已在 `.gitignore` 中排除），二次启动明显更快。

**常见问题**

- 首次打开页面较慢：正在拉取基金列表与历史行情，属正常现象。
- 个别基金无数据或指标为空：多为上游接口限流或该基金历史区间过短，稍后重试即可。
- 想清空本地缓存：删除 `.cache/` 目录后重启即可重建。

---

## 四、项目结构

```
.
├── app.py                 # Streamlit 主入口：11 个页面 + 全局样式 + 侧边栏
├── config.py              # 全局配置：常量、评分权重、风格画像、压力情景、配色
├── requirements.txt
├── fundquant/
│   ├── data.py            # 数据层：AKShare 拉取、双源降级、原子写入缓存
│   ├── indicators.py      # 指标计算（收益/风险/回撤/估值分位）
│   ├── scoring.py         # 四维评分、评级与指标白话释义
│   ├── predict.py         # LightGBM 多周期概率预测
│   ├── factors.py         # 因子诊断（IC、分层、相关性）
│   ├── holdings.py        # 持仓穿透与行业映射
│   ├── portfolio.py       # 组合优化与风险贡献
│   ├── allocation.py      # 大类资产配置与再平衡
│   ├── dca.py             # 定投测算
│   ├── backtest.py        # 策略回测对比
│   ├── stress.py          # 压力测试（历史情景 + 蒙特卡洛）
│   ├── charts.py          # Plotly 图表统一封装与配色
│   └── report.py          # HTML 报告生成（单基金 + 多基金/组合导出）
└── tests/                 # 指标、因子、组合、预测的单元测试
```

---

## 五、关键配置

全部集中在 `config.py`：

| 配置项 | 说明 |
|---|---|
| `RISK_FREE_RATE` / `TRADING_DAYS` / `COST_RATE` | 无风险利率、年交易日、单边交易成本 |
| `SCORE_WEIGHTS` | 四维评分权重 |
| `STYLE_PROFILES` | 保守 / 稳健 / 进取 三类风格的风险约束 |
| `STRESS_SCENARIOS` | 压力测试情景（区间、说明） |
| `CACHE_TTL` / `CACHE_VERSION` | 分级缓存有效期与口径版本 |
| `BENCHMARKS` / `ASSET_CLASS_ETFS` | 基准与跨资产代理 ETF |

---

## 六、测试

```bash
python -m pytest tests -q
```

---

## 七、自行部署（可选）

本项目已在 Streamlit Community Cloud 上线，无需自行部署。如果你希望部署一份独立副本：

1. Fork 本仓库（或把代码推到你自己的公开仓库；Community Cloud 免费版要求仓库为 public）。
2. 打开 <https://share.streamlit.io>，用 GitHub 账号登录并授权。
3. **New app** → 选择仓库与分支 `main` → **Main file path** 填 `app.py` → **Deploy**。
4. 构建完成后即可通过分配到的 `https://<app-name>.streamlit.app` 访问。

无需配置环境变量或密钥，`requirements.txt` 已声明全部依赖；首次启动会自动拉取并缓存行情数据。

---

## 八、数据来源与免责声明

- 数据来源：AKShare 聚合的东方财富 / 新浪 / 乐咕乐股等公开接口，存在延迟、缺失与接口变更风险。
- 所有回测与预测结果基于历史数据，**历史表现不代表未来收益**。
- 本项目仅供学习与研究使用，不构成任何投资建议；据此操作风险自担。

---

## 九、许可证

代码以 [MIT License](LICENSE) 授权，可自由使用、修改与分发（保留版权声明）。

请注意：**MIT 仅授权代码本身**，不代表对数据准确性、模型输出或任何投资结果作出担保；使用本项目产生的任何投资决策与后果由使用者自行承担。
