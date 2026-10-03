# -*- coding: utf-8 -*-
"""可视化图表库：基于 Plotly，统一极简金融专业配色，全部支持 hover/缩放/导出。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

C = config.COLORS
_THEME = None  # 深色模式激活时指向 DARK_COLORS，否则为 None


def set_palette(dark: bool = False):
    """切换全局调色板（浅色/深色），使所有图表函数与 _base 统一取色。"""
    global C, _THEME
    C = config.DARK_COLORS if dark else config.COLORS
    _THEME = C if dark else None


def _base(fig, title="", height=380, showlegend=None, colors=None):
    pal = colors or C
    is_dark = _THEME is not None or colors is not None
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=15, color=pal["primary"])),
        template="plotly_dark" if is_dark else "plotly_white",
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=pal["bg"] if is_dark else "#ffffff",
        hovermode="x unified",
        hoverlabel=dict(bgcolor=pal["primary"], font_color="#ffffff", font_size=12),
        font=dict(color=pal["neutral"], size=12),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="center", x=0.5,
                    bgcolor="rgba(30,41,59,0.7)" if is_dark else "rgba(255,255,255,0.7)",
                    font=dict(size=10, color=pal["neutral"])),
        dragmode="zoom",
        modebar=dict(bgcolor="rgba(255,255,255,0.08)", color=pal["neutral"]),
    )
    fig.update_xaxes(showgrid=True, gridcolor=pal["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=pal["grid"], zeroline=False)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def candlestick(hist: pd.DataFrame, name: str = ""):
    """K线图。"""
    h = hist.tail(180)
    fig = go.Figure(go.Candlestick(
        x=h["date"], open=h["open"], high=h["high"], low=h["low"], close=h["close"],
        increasing_line_color=C["up"], decreasing_line_color=C["down"],
        name=name or "价格"))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06),
                     type="date")
    return _base(fig, f"{name} 近180日K线", height=430, showlegend=False)


def price_line(prices_dict: dict, title="净值走势对比", normalize=True):
    """多资产价格/净值曲线（可归一化）。"""
    fig = go.Figure()
    for i, (name, s) in enumerate(prices_dict.items()):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if normalize:
            s = s / s.iloc[0]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 name=name, line=dict(color=C["series"][i % len(C["series"])], width=2)))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06),
                     type="date")
    return _base(fig, title, height=400)


def drawdown_chart(prices_or_nav: pd.Series, title="回撤走势"):
    """回撤曲线。"""
    s = pd.to_numeric(prices_or_nav, errors="coerce").dropna()
    dd = (s / s.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                               line=dict(color=C["down"], width=1.5), name="回撤"))
    fig.add_hline(y=0, line_color=C["ref_line"], line_width=1)
    fig.update_yaxes(title="回撤 (%)")
    return _base(fig, title, showlegend=False)


def radar_chart(metrics: dict, title="风险收益雷达"):
    """四维评分雷达。"""
    dims = ["收益", "风险", "估值", "稳定性"]
    vals = []
    for d in dims:
        if metrics and d in metrics:
            vals.append(metrics[d]["score"])
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=dims + [dims[0]], fill="toself",
        line=dict(color=C["accent"]), name="评分"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
                      showlegend=False, height=320, margin=dict(l=50, r=50, t=40, b=40))
    return fig


def radar_overlay(funds: dict, title="多基金评分雷达对比"):
    """多基金四维评分雷达叠加对比。

    funds: {基金名: breakdown}，breakdown 为 {维度: {"score": float}}。
    """
    dims = ["收益", "风险", "估值", "稳定性"]
    fig = go.Figure()
    for i, (name, breakdown) in enumerate(funds.items()):
        vals = [breakdown[d]["score"] for d in dims if d in breakdown]
        if len(vals) != len(dims):
            continue
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]], name=name,
            line=dict(color=C["series"][i % len(C["series"])], width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], visible=True)))
    fig = _base(fig, title, height=430)
    # 标题置顶；图例下移到底部，避免与标题重合，并预留足够底部边距
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=60, b=90))
    return fig


def pie_chart(weights: dict, title="组合权重分布"):
    """权重饼图（名称放底部图例，避免扇区标签重叠）。"""
    w = {k: v for k, v in weights.items() if v > 0}
    fig = go.Figure(go.Pie(
        labels=list(w.keys()), values=list(w.values()),
        hole=0.42, textinfo="percent", textposition="auto",
        hovertemplate="%{label}: %{percent}<extra></extra>",
        marker=dict(colors=C["series"])))
    fig = _base(fig, title, height=400, showlegend=True)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(l=20, r=20, t=60, b=70))
    return fig


def smile_chart(smile: pd.DataFrame, title="定投微笑曲线"):
    """定投成本 vs 市值曲线。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=smile.index, y=smile["累计投入"], name="累计投入",
                             line=dict(color=C["primary"], width=2)))
    fig.add_trace(go.Scatter(x=smile.index, y=smile["当前市值"], name="当前市值",
                             line=dict(color=C["up"], width=2)))
    fig.add_trace(go.Scatter(x=smile.index, y=smile["盈亏"], name="盈亏",
                             line=dict(color=C["accent"], width=1.5, dash="dot")))
    fig.update_yaxes(title="金额（元）")
    return _base(fig, title)


def stress_bar(stress: dict, title="压力测试情景对比"):
    """压力测试回撤条形图。"""
    names = list(stress.keys())
    mdd = [stress[n].get("最大回撤") for n in names]
    mdd = [v if isinstance(v, (int, float)) else None for v in mdd]
    fig = go.Figure(go.Bar(x=names, y=mdd, marker_color=C["down"],
                           text=[f"{v}%" if v is not None else "N/A" for v in mdd],
                           textposition="outside", name="最大回撤"))
    fig.update_yaxes(title="最大回撤 (%)")
    return _base(fig, title, showlegend=False)


def stress_drawdown_overlay(navs: dict, title="各情景回撤路径叠加"):
    """压力测试各情景回撤路径曲线叠加。

    将各情景的归一化净值转化为回撤百分比，叠加展示，便于观察不同情景下
    回撤深度与恢复速度的差异。
    """
    fig = go.Figure()
    for i, (name, nav) in enumerate(navs.items()):
        if not isinstance(nav, pd.Series):
            continue
        nav = pd.to_numeric(nav, errors="coerce").dropna()
        if len(nav) < 2:
            continue
        dd = (nav / nav.cummax() - 1) * 100
        color = C["series"][i % len(C["series"])]
        fig.add_trace(go.Scatter(
            x=list(range(len(dd))), y=dd.values, mode="lines", name=name,
            line=dict(color=color, width=1.6),
            hovertemplate="交易日 %{x}<br>回撤 %{y:.2f}%<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=C["ref_line"], line_width=1, line_dash="dash")
    fig.update_xaxes(title="情景内交易日")
    fig.update_yaxes(title="回撤 (%)")
    fig.update_layout(hovermode="x unified")
    return _base(fig, title, height=400)


def frontier_chart(frontier: pd.DataFrame, stats: dict = None, title="有效前沿"):
    """马科维茨有效前沿。"""
    fig = go.Figure()
    if frontier is not None and len(frontier):
        fig.add_trace(go.Scatter(x=frontier["vol"] * 100, y=frontier["ret"] * 100,
                                 mode="lines+markers", name="有效前沿",
                                 line=dict(color=C["accent"], width=2)))
    fig.update_xaxes(title="年化波动率 (%)")
    fig.update_yaxes(title="年化收益 (%)")
    return _base(fig, title, showlegend=False)


def bar_compare(data: dict, title="收益对比 (%)"):
    """横向对比条形图。"""
    fig = go.Figure(go.Bar(x=list(data.keys()), y=list(data.values()),
                           marker_color=C["series"][:len(data)], textposition="outside"))
    return _base(fig, title, showlegend=False)


def drawdown_compare(navs: dict, title="各策略回撤对比"):
    """多策略归一化净值的回撤曲线叠加。"""
    fig = go.Figure()
    for i, (name, nav) in enumerate(navs.items()):
        nav = pd.to_numeric(nav, errors="coerce").dropna()
        if len(nav) < 2:
            continue
        dd = (nav / nav.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name,
                                 line=dict(color=C["series"][i % len(C["series"])], width=1.6)))
    fig.add_hline(y=0, line_color=C["ref_line"], line_width=1)
    fig.update_yaxes(title="回撤 (%)")
    return _base(fig, title, height=400)


def probability_gauge(prob_up: float, title="上涨概率"):
    """涨跌概率仪表盘。"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_up * 100,
        number=dict(suffix="%", font=dict(size=36, color=C["primary"])),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=C["accent"]),
                   steps=[dict(range=[0, 50], color=C["down_bg"]),
                          dict(range=[50, 100], color=C["up_bg"])],
                   threshold=dict(line=dict(color=C["down"], width=3), thickness=0.75, value=50)),
        title=dict(text=title)))
    fig.update_layout(height=260, margin=dict(l=40, r=40, t=60, b=20))
    return fig


def corr_heatmap(corr: pd.DataFrame, title="成分相关性矩阵"):
    """相关性矩阵热力图（蓝-白-红发散：蓝=负相关，红=正相关，白=不相关）。"""
    labels = [str(x) for x in corr.columns]
    z = corr.values.astype(float)
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0.0, C["accent"]], [0.5, C["card_bg"]], [1.0, C["up"]]],
        text=text, texttemplate="%{text}", textfont=dict(size=11, color=C["text"]),
        colorbar=dict(title="相关系数", thickness=14),
        hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, title, height=440, showlegend=False)


def risk_contribution_bar(rc_df: pd.DataFrame, title="风险贡献分解"):
    """风险贡献分解图：权重 vs 风险贡献占比 并列对比，识别风险集中资产。"""
    x = rc_df["资产"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=rc_df["权重"], name="权重 (%)",
                         marker_color=C["accent"],
                         text=[f"{v:.1f}%" for v in rc_df["权重"]],
                         textposition="outside"))
    fig.add_trace(go.Bar(x=x, y=rc_df["风险贡献占比(%)"], name="风险贡献占比 (%)",
                         marker_color=C["down"],
                         text=[f"{v:.1f}%" for v in rc_df["风险贡献占比(%)"]],
                         textposition="outside"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(title="%", rangemode="tozero")
    return _base(fig, title, height=380)


def rolling_line(series: pd.Series, title="", color=None, fill=False):
    """单序列折线图（用于滚动收益 / 滚动波动等拆分展示，避免上下堆叠不协调）。"""
    color = color or C["primary"]
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines",
                               name=title, line=dict(color=color, width=1.8),
                               fill="tozeroy" if fill else None))
    if fill:
        fig.update_traces(fillcolor="rgba(43,108,176,0.12)")
    return _base(fig, title=title, height=360, showlegend=False)


def risk_return_scatter(points: pd.DataFrame, title="收益-风险散点"):
    """滚动子窗口的收益-风险散点（x=年化波动, y=年化收益），末点高亮为当前。"""
    fig = go.Figure()
    if points is not None and len(points):
        x = points["年化波动"].values
        y = points["年化收益"].values
        fig.add_trace(go.Scatter(x=x[:-1], y=y[:-1], mode="markers", name="历史窗口",
                                 marker=dict(color=C["neutral"], size=7, opacity=0.55),
                                 hovertemplate="波动 %{x:.1f}% · 收益 %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers+text", name="当前",
                                 text=["当前"], textposition="top center",
                                 marker=dict(color=C["accent"], size=13,
                                             line=dict(width=2, color="#ffffff")),
                                 hovertemplate="波动 %{x:.1f}% · 收益 %{y:.1f}%<extra>当前</extra>"))
    fig.update_xaxes(title="年化波动率 (%)")
    fig.update_yaxes(title="年化收益 (%)")
    fig.add_hline(y=0, line_color=C["ref_line"], line_width=1)
    return _base(fig, title, height=380, showlegend=True)


def factor_ic_chart(frame: pd.DataFrame, title="", cumulative=False):
    """因子 IC 时序 / 累计 IC 多序列折线（横轴日期，纵轴 IC）。"""
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        s = frame[col]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=col,
                                 line=dict(color=C["series"][i % len(C["series"])], width=1.6)))
    fig.add_hline(y=0, line_color=C["ref_line"], line_width=1)
    fig.update_yaxes(title="累计 IC" if cumulative else "IC")
    return _base(fig, title=title, height=400)


def reliability_diagram(calib_data: dict, title="概率校准曲线 (Reliability Diagram)"):
    """概率校准曲线：展示模型预测概率与实际发生频率的一致性。

    calib_data: calibration_curve_data() 返回的 {bin_label: {pred_mean, actual_freq, count}}。
    """
    labels = list(calib_data.keys())
    pred_means = [calib_data[k]["pred_mean"] for k in labels]
    actual_freqs = [calib_data[k]["actual_freq"] for k in labels]
    counts = [calib_data[k]["count"] for k in labels]
    sizes = [max(6, c * 0.8) for c in counts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=C["ref_line"], width=1.5, dash="dash"),
                             name="完美校准 (对角线)", showlegend=True))
    fig.add_trace(go.Scatter(x=pred_means, y=actual_freqs,
                             mode="markers+text",
                             marker=dict(size=sizes, color=C["accent"],
                                         line=dict(color="#1f2937", width=1)),
                             text=[f"n={c}" for c in counts],
                             textposition="top center",
                             textfont=dict(size=10, color=C["neutral"]),
                             name="实际观测", showlegend=False))
    fig.update_xaxes(title="预测概率", range=[-0.05, 1.05], dtick=0.1)
    fig.update_yaxes(title="实际发生频率", range=[-0.05, 1.05], dtick=0.1)
    fig.add_annotation(x=0.95, y=0.05, text="高估", showarrow=False,
                       font=dict(size=12, color=C["text_secondary"]))
    fig.add_annotation(x=0.05, y=0.95, text="低估", showarrow=False,
                       font=dict(size=12, color=C["text_secondary"]))
    return _base(fig, title, height=400, showlegend=True)