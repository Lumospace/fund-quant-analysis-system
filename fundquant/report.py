# -*- coding: utf-8 -*-
"""智能配置建议输出（落地应用出口）与报告生成。

分层输出操作建议：优势/风险 -> 走势概率判断 -> 适配人群 -> 操作指令 ->
仓位比例 -> 止盈止损点 -> 持仓注意事项 -> 压力测试结论。
"""
import numpy as np
import pandas as pd
import config
from .scoring import GLOSSARY
from . import indicators as ind


def generate_operation_advice(metrics: dict, prediction: dict = None,
                              style: str = "稳健", trend: dict = None,
                              stress: dict = None, score: dict = None) -> dict:
    """生成完整分层操作建议。"""
    prof = config.STYLE_PROFILES.get(style, config.STYLE_PROFILES["稳健"])
    vol = metrics.get("年化波动率", 0) / 100
    mdd = abs(metrics.get("最大回撤", 0)) / 100
    ann = metrics.get("年化收益", 0) / 100

    # 操作指令
    if mdd > prof["max_drawdown_limit"] * 1.5:
        action = "减仓/清仓"
    elif vol > prof["target_vol"] * 1.3:
        action = "减仓"
    elif ann > 0 and mdd <= prof["max_drawdown_limit"] * 0.7:
        action = "加仓"
    else:
        action = "持有"

    # 概率判断
    prob_txt = "暂无 AI 预测"
    prob_up = None
    if prediction:
        prob_up = prediction.get("prob_up")
        if prob_up is not None:
            h = prediction.get("horizon", 20)
            prob_txt = (f"未来 {h} 日上涨概率 {prob_up*100:.0f}%、"
                        f"下跌概率 {(1-prob_up)*100:.0f}%（置信度 {prediction.get('confidence', 0.3)*100:.0f}%）")

    # 仓位比例（按风格与风险等级）
    risk_load = max(0.1, 1.0 - mdd / max(prof["max_drawdown_limit"] * 2, 0.2))
    position = np.clip(risk_load, 0.15, 1.0)
    pos_txt = f"参考仓位 {position*100:.0f}%（相对可用资金）"

    # 止盈止损点（基于近1年波动给出参考）
    stop_loss = -max(0.05, vol * 1.5)
    take_profit = max(0.10, vol * 2.5)
    point_txt = (f"参考止盈 +{take_profit*100:.0f}%、止损 -{abs(stop_loss)*100:.0f}%"
                 f"（基于年化波动 {vol*100:.1f}% 推得，可自行调整）")

    # 压力测试结论
    stress_txt = "暂无压力测试"
    if stress:
        worst = max([v["最大回撤"] for v in stress.values()
                     if isinstance(v.get("最大回撤"), (int, float))], default=None)
        if worst is not None:
            stress_txt = f"历史极端情景下最大回撤约 {worst:.1f}%，请确认可承受该幅度缩水"

    # 适配人群
    fit = {
        "保守": "适合看重本金安全、厌恶回撤的投资者",
        "稳健": "适合追求收益与波动平衡的长期投资者",
        "进取": "适合能承受较大波动、追求高弹性的投资者",
    }.get(style, "")

    return {
        "核心优势与风险": (f"年化收益 {metrics.get('年化收益', '--')}%、年化波动 {vol*100:.1f}%、"
                     f"最大回撤 {metrics.get('最大回撤', '--')}%。"),
        "走势概率判断": prob_txt,
        "适配人群": fit,
        "操作指令": action,
        "仓位比例": pos_txt,
        "止盈止损": point_txt,
        "注意事项": (
            f"投资周期宜匹配 {style} 风格；回测存在幸存者偏差；"
            f"AI 预测仅为概率参考，不保证涨跌。"
        ),
        "压力测试结论": stress_txt,
    }


def metrics_markdown_table(metrics: dict) -> str:
    """指标 -> Markdown 表格 + 白话释义。"""
    rows = []
    for k, v in metrics.items():
        gloss = GLOSSARY.get(k, "")
        rows.append(f"| {k} | {v} | {gloss} |")
    header = "| 指标 | 数值 | 白话释义 |\n|---|---|---|\n"
    return header + "\n".join(rows)


def _fmt_pct(v: float, scale: float = 100.0) -> str:
    """数值转百分比字符串，非数值返回占位符。"""
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


def build_fund_report_html(name: str, code: str, metrics: dict, score: dict,
                           advice: dict, forecasts: dict = None, holdings=None,
                           stress: dict = None) -> str:
    """生成单基金量化分析报告 HTML（自包含，可导出）。

    板块顺序：①基本信息 ②重仓股 ③综合评分 ④四维度评分明细
    ⑤核心指标 ⑥AI 多周期预测 ⑦压力测试四情景 ⑧落地操作建议 ⑨合规声明。
    """
    score_rows = "".join(
        f"<tr><td>{k}</td><td>{d['weight']*100:.0f}%</td>"
        f"<td>{d['score']:.1f}</td><td>{'; '.join(d['reasons'])}</td></tr>"
        for k, d in score["breakdown"].items()
    )
    advice_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in advice.items()
    )
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>{name} 量化分析报告</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:32px;color:#1e293b;background:#f5f7fa}}
h1{{color:#1E3A5F}} h2{{color:#1E3A5F;border-bottom:2px solid #1E3A5F;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;background:#fff}}
td,th{{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;font-size:13px}}
th{{background:#1E3A5F;color:#fff}}
.badge{{display:inline-block;background:#18A058;color:#fff;padding:4px 14px;border-radius:16px;font-size:16px}}
.warn{{background:#FFF1F0;border-left:4px solid #E5484D;padding:10px 14px;margin:14px 0}}
</style></head><body>
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
<h2>三、综合评分</h2>
<p style="font-size:18px; font-weight:600; color:#1E3A5F; margin:6px 0;">综合评分 {score['total']} · {score['rating']}</p>
<h2>四、四维度评分明细</h2>
<table><tr><th>维度</th><th>权重</th><th>得分</th><th>打分依据</th></tr>{score_rows}</table>
<h2>五、核心指标</h2>
{metrics_markdown_html(metrics)}
<h2>六、AI 多周期预测</h2>
<table><tr><th>周期</th><th>上涨概率</th><th>下跌概率</th><th>置信度</th><th>预期收益</th><th>收益区间(10%~90%)</th></tr>{_forecast_rows(forecasts)}</table>
<h2>七、压力测试四情景</h2>
<table><tr><th>情景</th><th>最大回撤</th><th>缩水幅度</th><th>年化波动率</th><th>修复周期</th><th>说明</th></tr>{_stress_rows(stress)}</table>
<h2>八、落地操作建议</h2>
<table><tr><th>项目</th><th>内容</th></tr>{advice_rows}</table>
<h2>九、合规声明</h2>
<div class="warn">{config.RISK_DISCLAIMER}</div>
</body></html>"""
    return html


def metrics_markdown_html(metrics: dict) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{GLOSSARY.get(k,'')}</td></tr>"
                   for k, v in metrics.items())
    return "<table><tr><th>指标</th><th>数值</th><th>白话释义</th></tr>" + rows + "</table>"