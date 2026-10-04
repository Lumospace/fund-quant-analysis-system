# -*- coding: utf-8 -*-
"""量化分析报告生成。"""
import pandas as pd
import config
from .scoring import GLOSSARY
from . import indicators as ind


_HTML_HEAD = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:32px;color:#1e293b;background:#f5f7fa}}
h1{{color:#1E3A5F}} h2{{color:#1E3A5F;border-bottom:2px solid #1E3A5F;padding-bottom:6px;margin-top:24px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;background:#fff}}
td,th{{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;font-size:13px}}
th{{background:#1E3A5F;color:#fff}}
.badge{{display:inline-block;background:#18A058;color:#fff;padding:4px 14px;border-radius:16px;font-size:16px}}
.warn{{background:#FFF1F0;border-left:4px solid #E5484D;padding:10px 14px;margin:14px 0}}
.footer{{margin-top:32px;font-size:12px;color:#94a3b8;border-top:1px solid #e2e8f0;padding-top:12px}}
</style></head><body>"""

# 深色版报告样式：供 App 内联预览在深色模式下嵌入，避免白底表格突兀；
# 下载导出的报告仍用 _HTML_HEAD（浅色，便于打印）。
_HTML_HEAD_DARK = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:32px;color:#F1F5F9;background:#0F172A}}
h1{{color:#8BB8E8}} h2{{color:#8BB8E8;border-bottom:2px solid #8BB8E8;padding-bottom:6px;margin-top:24px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;background:#1E293B}}
td,th{{border:1px solid #334155;padding:8px 10px;text-align:left;font-size:13px;color:#F1F5F9}}
th{{background:#8BB8E8;color:#0F172A}}
.badge{{display:inline-block;background:#18A058;color:#fff;padding:4px 14px;border-radius:16px;font-size:16px}}
.warn{{background:#3B1A1A;border-left:4px solid #FC8181;padding:10px 14px;margin:14px 0;color:#F1F5F9}}
.footer{{margin-top:32px;font-size:12px;color:#94A3B8;border-top:1px solid #334155;padding-top:12px}}
</style></head><body>"""

_HTML_FOOT = '<div class="footer">本报告由 FundQuant 量化分析系统自动生成，仅供参考，不构成投资建议。</div></body></html>'


def metrics_markdown_table(metrics: dict) -> str:
    rows = []
    for k, v in metrics.items():
        gloss = GLOSSARY.get(k, "")
        rows.append(f"| {k} | {v} | {gloss} |")
    header = "| 指标 | 数值 | 白话释义 |\n|---|---|---|\n"
    return header + "\n".join(rows)


def _fmt_pct(v: float, scale: float = 100.0) -> str:
    if isinstance(v, (int, float)):
        return f"{v * scale:.1f}%"
    return "—"


def _forecast_rows(forecasts: dict) -> str:
    if not forecasts:
        return "<tr><td colspan='6'>暂无多周期预测数据</td></tr>"
    rows = []
    for h, f in sorted(forecasts.items()):
        lo, hi = f.get("return_range", (None, None))
        rows.append(
            f"<tr><td>{h} 日</td>"
            f"<td>{_fmt_pct(f.get('prob_up'))}</td>"
            f"<td>{_fmt_pct(f.get('prob_down'))}</td>"
            f"<td>{_fmt_pct(f.get('confidence'))}</td>"
            f"<td>{_fmt_pct(f.get('expected_return', None), scale=1.0)}</td>"
            f"<td>{_fmt_pct(lo, scale=1.0)} ~ {_fmt_pct(hi, scale=1.0)}</td></tr>"
        )
    return "".join(rows)


def _stress_rows(stress: dict) -> str:
    if not stress:
        return "<tr><td colspan='6'>暂无压力测试数据</td></tr>"
    rows = []
    for name, v in stress.items():
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{_fmt_pct(v.get('最大回撤'), scale=1.0)}</td>"
            f"<td>{_fmt_pct(v.get('缩水幅度'), scale=1.0)}</td>"
            f"<td>{_fmt_pct(v.get('年化波动率'), scale=1.0)}</td>"
            f"<td>{v.get('修复周期(交易日)', '—')}</td>"
            f"<td>{v.get('说明', '')}</td></tr>"
        )
    return "".join(rows)


def _holdings_table(holdings) -> str:
    if holdings is None or holdings.empty:
        return "<table><tr><th>重仓股</th></tr><tr><td>暂无重仓股数据</td></tr></table>"
    cols = [c for c in ["序号", "股票代码", "股票名称", "占净值比例", "持股数", "持仓市值", "季度"]
            if c in holdings.columns]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = []
    for _, r in holdings.head(10).iterrows():
        body.append("<tr>" + "".join(
            f"<td>{'' if v is None else v}</td>" for v in r[cols]) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _lookthrough_section(lookthrough: dict) -> str:
    """持仓穿透章节：前十大重仓的行业/规模/估值风格暴露 + 集中度。"""
    if not lookthrough or float(lookthrough.get("concentration", 0) or 0) <= 0:
        return "<p>暂无持仓穿透数据（该基金可能为指数/债基或未披露前十大重仓）。</p>"
    conc = float(lookthrough.get("concentration", 0.0))

    def _bar(s: pd.Series, top: int) -> str:
        if s is None or s.empty:
            return "<p>—</p>"
        items = "".join(f"<tr><td>{k}</td><td>{v:.1f}%</td></tr>"
                        for k, v in s.head(top).items())
        return f"<table><tr><th>分类</th><th>占比</th></tr>{items}</table>"

    ind_part = lookthrough.get("industry")
    size_part = lookthrough.get("size")
    value_part = lookthrough.get("value")
    return (
        f"<p>前十大重仓合计占净值 <strong>{conc:.1f}%</strong>，其底层结构穿透如下。</p>"
        f"<p><strong>行业分布（前6）：</strong></p>{_bar(ind_part, 6)}"
        f"<p><strong>规模风格（大盘/中盘/小盘）：</strong></p>{_bar(size_part, 4)}"
        f"<p><strong>估值风格（价值/平衡/成长）：</strong></p>{_bar(value_part, 3)}"
    )


def _factors_section(factors) -> str:
    """因子有效性诊断章节：有效/预警/下线计数 + 最强有效因子 + 明细表。"""
    if factors is None or len(factors) == 0:
        return "<p>暂无因子诊断数据。</p>"
    df = factors
    state = df["状态"].value_counts()
    valid = df[df["状态"] == "有效"]
    top = ""
    if not valid.empty:
        t = valid.sort_values("IC", key=lambda s: s.abs(), ascending=False).iloc[0]
        top = (f"最强有效因子为 <strong>{t['因子']}</strong>"
               f"（|IC|={abs(t['IC']):.3f}，{t['方向']}）。")
    cnt = "、".join(f"{k} {v} 个" for k, v in state.items()) or "无"
    head = "".join(f"<th>{c}</th>" for c in ["因子", "类别", "IC", "IR", "t值", "状态"])
    rows = []
    for _, r in df.iterrows():
        ic = "—" if pd.isna(r["IC"]) else f"{r['IC']:.3f}"
        ir = "—" if pd.isna(r["IR"]) else f"{r['IR']:.3f}"
        tv = "—" if pd.isna(r.get("t值")) else f"{r['t值']:.2f}"
        rows.append(f"<tr><td>{r['因子']}</td><td>{r['类别']}</td>"
                    f"<td>{ic}</td><td>{ir}</td><td>{tv}</td><td>{r['状态']}</td></tr>")
    table = f"<table><tr>{head}</tr>{''.join(rows)}</table>"
    return (f"<p>因子自动校验（滚动 IC 失效占比）：{cnt}。{top}"
            f"「已下线」因子已从预测特征中剔除。</p>{table}")


def _strategies_section(summary) -> str:
    """策略回测对比章节：五套策略统一指标表 + 夏普最优结论。"""
    if summary is None or summary.empty:
        return "<p>暂无策略回测数据。</p>"
    df = summary.copy()
    pct_cols = {"总收益", "年化收益", "年化波动", "最大回撤"}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (int, float)) and not pd.isna(v):
                cells.append(f"<td>{v*100:.1f}%</td>" if c in pct_cols
                             else f"<td>{v:.2f}</td>")
            else:
                cells.append("<td>—</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = f"<table><tr>{head}</tr>{''.join(rows)}</table>"
    sh = df[df["夏普比率"].notna()]
    concl = ""
    if not sh.empty:
        b = sh.loc[sh["夏普比率"].idxmax()]
        concl = (f"<p>五套策略横向对比（择时类已计入单边手续费+滑点）："
                 f"夏普比率最高为 <strong>{b['策略']}</strong>"
                 f"（夏普 {b['夏普比率']:.2f}、年化 {b['年化收益']*100:.1f}%、"
                 f"最大回撤 {b['最大回撤']*100:.1f}%）。"
                 f"择时/定投策略的换手成本已扣除，与买入持有裸收益不可直接比较。</p>")
    return table + concl


def _dca_section(dca: dict) -> str:
    """定投测算章节：核心结论 + 时点优选 + 三止盈策略对比。"""
    if not dca or not dca.get("backtest"):
        return "<p>暂无定投测算数据（样本不足）。</p>"
    b = dca["backtest"]
    freq = b.get("freq", "月")
    amt = b.get("每期金额", 1000)
    start = dca.get("start", "")
    irr = b.get("定投IRR")
    irr_txt = f"{irr:.1f}%" if isinstance(irr, (int, float)) else "—"
    concl = (f"<p>按<b>每{freq}定额 {amt:g} 元</b>定投{f'（{start} 起）' if start else ''}："
             f"累计投入 <strong>{b['总投入']:,.0f} 元</strong>、"
             f"期末资产 <strong>{b['期末资产']:,.0f} 元</strong>，"
             f"累计收益率 <strong>{b['收益率']}%</strong>、"
             f"年化 IRR <strong>{irr_txt}</strong>、"
             f"期间最大回撤 <strong>{b['最大回撤']}%</strong>"
             f"（共定投 {b['定投期数']} 期）。</p>")

    timing_txt = ""
    timing = dca.get("timing") or {}
    if timing:
        vp = timing.get("估值分位")
        vp_txt = f"{vp:.0f}%" if isinstance(vp, (int, float)) else "—"
        timing_txt = (f"<p>时点优选：当前估值分位 <strong>{vp_txt}</strong> → "
                      f"<strong>{timing.get('建议', '')}</strong>。</p>")

    stop_txt = ""
    sdf = (dca.get("stop") or {}).get("止盈对比")
    if sdf is not None and not sdf.empty:
        head = "".join(f"<th>{c}</th>" for c in ["策略", "参数", "期末资产", "模拟收益率"])
        rows = "".join(f"<tr><td>{r['策略']}</td><td>{r['参数']}</td>"
                       f"<td>{r['期末资产']:,.0f}</td><td>{r['模拟收益率']:.1f}%</td></tr>"
                       for _, r in sdf.iterrows())
        stop_txt = (f"<p>三种止盈策略模拟对比（止盈后清仓不再定投）：</p>"
                    f"<table><tr>{head}</tr>{rows}</table>")
    return concl + timing_txt + stop_txt


def build_fund_report_html(name: str, code: str, metrics: dict, score: dict,
                           forecasts: dict = None, holdings=None,
                           stress: dict = None, strategies=None, dca: dict = None,
                           lookthrough: dict = None, factors=None,
                           dark: bool = False) -> str:
    score_rows = "".join(
        f"<tr><td>{k}</td><td>{d['weight']*100:.0f}%</td>"
        f"<td>{d['score']:.1f}</td><td>{'; '.join(d['reasons'])}</td></tr>"
        for k, d in score["breakdown"].items()
    )
    head = (_HTML_HEAD_DARK if dark else _HTML_HEAD).format(title=f"{name} 量化分析报告")
    score_color = "#8BB8E8" if dark else "#1E3A5F"
    html = f"""{head}
<h1>{name}（{code}）量化分析报告</h1>
<h2>一、基本信息</h2>
<table>
<tr><th>项目</th><th>内容</th></tr>
<tr><td>基金名称</td><td>{name}</td></tr>
<tr><td>基金代码</td><td>{code}</td></tr>
<tr><td>投资风格</td><td>{score['style']}</td></tr>
<tr><td>生成时间</td><td>{pd.Timestamp.now():%Y-%m-%d %H:%M}</td></tr>
</table>
<h2>二、重仓股（前十）</h2>
{_holdings_table(holdings)}
<h2>三、持仓结构与风格</h2>
{_lookthrough_section(lookthrough)}
<h2>四、综合评分</h2>
<p style="font-size:18px; font-weight:600; color:{score_color}; margin:6px 0;">综合评分 {score['total']} · {score['rating']}</p>
<h2>五、四维度评分明细</h2>
<table><tr><th>维度</th><th>权重</th><th>得分</th><th>打分依据</th></tr>{score_rows}</table>
<h2>六、核心指标</h2>
{metrics_markdown_html(metrics)}
<h2>七、AI 多周期预测</h2>
<table><tr><th>周期</th><th>上涨概率</th><th>下跌概率</th><th>置信度</th><th>预期收益</th><th>收益区间(10%~90%)</th></tr>{_forecast_rows(forecasts)}</table>
<h2>八、因子有效性诊断</h2>
{_factors_section(factors)}
<h2>九、压力测试情景</h2>
<table><tr><th>情景</th><th>最大回撤</th><th>缩水幅度</th><th>年化波动率</th><th>修复周期</th><th>说明</th></tr>{_stress_rows(stress)}</table>
<h2>十、策略回测对比</h2>
{_strategies_section(strategies)}
<h2>十一、定投测算</h2>
{_dca_section(dca)}
<h2>十二、合规声明</h2>
<div class="warn">{config.RISK_DISCLAIMER}</div>
{_HTML_FOOT}"""
    return html


def metrics_markdown_html(metrics: dict) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{GLOSSARY.get(k,'')}</td></tr>"
                   for k, v in metrics.items())
    return "<table><tr><th>指标</th><th>数值</th><th>白话释义</th></tr>" + rows + "</table>"


def build_compare_report_html(names: list, metrics_map: dict, score_map: dict,
                               rating_map: dict, prices_map: dict = None,
                               breakdowns: dict = None) -> str:
    """生成多基金横向对比报告 HTML。"""
    st = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    compare_keys = ["年化收益", "年化波动率", "最大回撤", "夏普比率", "卡玛比率", "估值分位"]
    rows = []
    for k in compare_keys:
        cells = "".join(f"<td>{metrics_map[nm].get(k, '—'):.2f}</td>" if isinstance(metrics_map[nm].get(k), (int, float)) else f"<td>—</td>" for nm in names)
        gloss = GLOSSARY.get(k, "")
        rows.append(f"<tr><td>{k}</td><td>{gloss}</td>{cells}</tr>")
    compare_table = f"<table><tr><th>指标</th><th>释义</th>{''.join(f'<th>{nm}</th>' for nm in names)}</tr>{''.join(rows)}</table>"

    score_rows = "".join(
        f"<tr><td>{nm}</td><td>{score_map[nm]:.1f}</td><td>{rating_map[nm]}</td></tr>"
        for nm in names
    )

    if breakdowns:
        dims = list(next(iter(breakdowns.values())).keys())
        radar_rows = []
        for dim in dims:
            cells = "".join(f"<td>{breakdowns[nm][dim]['score']:.1f}</td>" for nm in names)
            radar_rows.append(f"<tr><td>{dim}</td>{cells}</tr>")

    rank_sorted = sorted(names, key=lambda nm: score_map[nm], reverse=True)
    best = rank_sorted[0]

    body = f"""
<h1>基金横向对比报告</h1>
<table><tr><th>项目</th><th>内容</th></tr>
<tr><td>对比基金</td><td>{'、'.join(names)}</td></tr>
<tr><td>最优评分</td><td>{best}（{score_map[best]:.1f} 分 · {rating_map[best]}）</td></tr>
<tr><td>生成时间</td><td>{st}</td></tr></table>

<h2>核心指标横向对比</h2>
{compare_table}

<h2>综合评分排名</h2>
<table><tr><th>基金</th><th>综合评分</th><th>评级</th></tr>{score_rows}</table>
"""

    if breakdowns:
        body += f"""
<h2>四维评分明细</h2>
<table><tr><th>维度</th>{''.join(f'<th>{nm}</th>' for nm in names)}</tr>{''.join(radar_rows)}</table>
"""

    def _v(nm, key):
        v = metrics_map[nm].get(key)
        return v if isinstance(v, (int, float)) else None

    def _arg(nms, key, mm):
        vs = [(nm, _v(nm, key)) for nm in nms if _v(nm, key) is not None]
        return (max if mm == "max" else min)(vs, key=lambda x: x[1]) if vs else None

    concl = [f"<p>在当前投资风格下，综合评分最高：<strong>{best}</strong>"
             f"（{rating_map[best]}，{score_map[best]:.1f} 分）。</p>"]
    r = _arg(names, "年化收益", "max")
    if r:
        concl.append(f"<p>收益维度：<strong>{r[0]}</strong> 年化收益最高（{r[1]:.2f}%）。</p>")
    md = _arg(names, "最大回撤", "max")
    mx = _arg(names, "最大回撤", "min")
    if md and mx and md[0] != mx[0]:
        concl.append(f"<p>风险维度：<strong>{md[0]}</strong> 回撤控制最好（最大回撤 {md[1]:.2f}%），"
                     f"<strong>{mx[0]}</strong> 回撤最深（{mx[1]:.2f}%）。</p>")
    v = _arg(names, "年化波动率", "min")
    if v:
        concl.append(f"<p><strong>{v[0]}</strong> 年化波动最低（{v[1]:.2f}%），持有体验更平稳。</p>")
    sh = _arg(names, "夏普比率", "max")
    if sh:
        concl.append(f"<p>风险调整后收益：<strong>{sh[0]}</strong> 夏普比率最高（{sh[1]:.2f}）。</p>")

    body += f"""
<h2>结论</h2>
{''.join(concl)}
"""
    return _HTML_HEAD.format(title="基金横向对比报告") + body + _HTML_FOOT


def _df_table(df: pd.DataFrame, pct_cols: set = None) -> str:
    """将 DataFrame 渲染为 HTML 表格（用于样本外/再平衡等补充对比）。"""
    if df is None or len(df) == 0:
        return "<p>无可用数据。</p>"
    pct_cols = pct_cols or set()
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (int, float)):
                if pd.isna(v):
                    cells.append("<td>—</td>")
                else:
                    suffix = "%" if c in pct_cols else ""
                    cells.append(f"<td>{v:.2f}{suffix}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def build_portfolio_report_html(opt: dict, ret: pd.DataFrame, prices_dict: dict,
                                 method: str, mode_label: str,
                                 walk_forward: pd.DataFrame = None) -> str:
    """生成组合优化结果报告 HTML（可选附带样本外回测对比）。"""
    st = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    w = opt["weights"]
    stats = opt.get("stats", {})

    w_rows = "".join(
        f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>"
        for k, v in sorted(w.items(), key=lambda x: -x[1])
    )
    ws = sum(v for v in w.values())
    w_note = f"（权重合计 {ws*100:.1f}%，{'满仓' if abs(ws-1)<0.01 else '有余仓'}）"

    stat_rows = ""
    for k, label in [("年化收益", "年化收益"), ("最大回撤", "最大回撤"),
                     ("夏普比率", "夏普比率"), ("年化波动率", "年化波动率"),
                     ("VaR95", "VaR95"), ("CVaR95", "CVaR95")]:
        if k in stats:
            v = stats[k]
            if isinstance(v, float):
                stat_rows += f"<tr><td>{label}</td><td>{v:.2f}%</td></tr>"
            else:
                stat_rows += f"<tr><td>{label}</td><td>{v}</td></tr>"

    sorted_names = [k for k, _ in sorted(w.items(), key=lambda x: -x[1])]

    body = f"""
<h1>投资组合量化优化报告</h1>
<table><tr><th>项目</th><th>内容</th></tr>
<tr><td>优化方法</td><td>{mode_label}</td></tr>
<tr><td>成分数量</td><td>{len(w)}</td></tr>
<tr><td>生成时间</td><td>{st}</td></tr></table>

<h2>最优权重</h2>
<table><tr><th>资产</th><th>权重</th></tr>{w_rows}</table>
<p>{w_note}</p>

<h2>组合绩效指标</h2>
<table><tr><th>指标</th><th>数值</th></tr>{stat_rows}</table>

<h2>成分资产相关性矩阵</h2>
{_corr_table(ret.corr() if ret.shape[1] > 1 else None, sorted_names)}

<h2>风险结构</h2>
<p>权重最高的资产未必是主要风险来源；需结合风险贡献占比综合判断。</p>
"""

    note = opt.get("note")
    if note:
        body += f'<div class="warn">{note}</div>'

    if walk_forward is not None and len(walk_forward):
        body += f"""
<h2>优化方法样本外回测对比</h2>
{_df_table(walk_forward, pct_cols={"年化收益", "年化波动率", "最大回撤", "VaR95", "CVaR95"})}
<p>上表按夏普比率降序排列，样本外表现最优的优化方法为 <strong>{walk_forward.iloc[0]['方法']}</strong>。</p>
"""

    return _HTML_HEAD.format(title="组合优化报告") + body + _HTML_FOOT


def _corr_table(corr, names) -> str:
    if corr is None or corr.empty:
        return "<p>无法计算相关性矩阵。</p>"
    head = "".join(f"<th>{c}</th>" for c in names)
    rows = []
    for n in names:
        cells = "".join(f"<td>{corr.loc[n, c]:.2f}</td>" if n in corr.index and c in corr.columns else "<td>—</td>" for c in names)
        rows.append(f"<tr><th>{n}</th>{cells}</tr>")
    return f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table>"


def build_allocation_report_html(method: str, weights: dict, stats: dict,
                                  prices_dict: dict = None,
                                  rebalance: pd.DataFrame = None) -> str:
    """生成大类资产配置结果报告 HTML（可选附带再平衡回测对比）。"""
    st = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    w_rows = "".join(
        f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>"
        for k, v in sorted(weights.items(), key=lambda x: -x[1])
    )
    stat_rows = ""
    for k, label in [("年化收益", "年化收益"), ("年化波动率", "年化波动率"),
                     ("最大回撤", "最大回撤"), ("夏普比率", "夏普比率")]:
        if k in stats:
            v = stats[k]
            if isinstance(v, float):
                stat_rows += f"<tr><td>{label}</td><td>{v*100:.2f}%</td></tr>"
            else:
                stat_rows += f"<tr><td>{label}</td><td>{v}</td></tr>"

    body = f"""
<h1>大类资产配置报告</h1>
<table><tr><th>项目</th><th>内容</th></tr>
<tr><td>配置方法</td><td>{method}</td></tr>
<tr><td>生成时间</td><td>{st}</td></tr></table>

<h2>最优权重</h2>
<table><tr><th>资产</th><th>权重</th></tr>{w_rows}</table>

<h2>组合绩效指标</h2>
<table><tr><th>指标</th><th>数值</th></tr>{stat_rows}</table>
"""
    if rebalance is not None and len(rebalance):
        body += f"""
<h2>再平衡回测（vs 买入持有）</h2>
{_df_table(rebalance, pct_cols={"年化收益", "年化波动率", "最大回撤", "VaR95", "CVaR95"})}
"""
    return _HTML_HEAD.format(title="大类资产配置报告") + body + _HTML_FOOT


def build_stress_report_html(name: str, res: dict, navs: dict = None) -> str:
    """生成压力测试报告 HTML（单基金或多基金组合）。"""
    st = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    body = f"""
<h1>压力测试报告</h1>
<table><tr><th>项目</th><th>内容</th></tr>
<tr><td>测试对象</td><td>{name}</td></tr>
<tr><td>情景数</td><td>{len(res)}（含历史回放 + 蒙特卡洛）</td></tr>
<tr><td>生成时间</td><td>{st}</td></tr></table>

<h2>各情景压力测试结果</h2>
<table><tr><th>情景</th><th>最大回撤</th><th>缩水幅度</th><th>年化波动率</th><th>修复周期</th><th>说明</th></tr>{_stress_rows(res)}</table>

<h2>结论</h2>
"""
    worst = max([v["最大回撤"] for v in res.values() if isinstance(v.get("最大回撤"), (int, float))], default=None)
    if worst is not None:
        body += f"<p>历史极端情景下最大回撤约 {worst:.1f}%，请确认可承受该幅度缩水。</p>"
    else:
        body += "<p>压力测试数据不足，无法给出定量结论。</p>"

    return _HTML_HEAD.format(title="压力测试报告") + body + _HTML_FOOT