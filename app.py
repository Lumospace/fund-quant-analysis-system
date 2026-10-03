# -*- coding: utf-8 -*-
"""基金投资量化分析系统 — Streamlit 主入口。

运行：streamlit run app.py
模块化驾驶舱：顶部导航 + 侧边菜单（含场内/场外全局筛选）+ 卡片主区，全功能页面。
"""
import datetime as _dt

import pandas as pd
import streamlit as st

import config
from fundquant import data, indicators as ind, factors as fac, scoring, predict
from fundquant import portfolio as pf, stress, dca, report, charts, backtest, allocation, holdings

st.set_page_config(page_title="基金投资量化分析系统", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

RISK = config.RISK_DISCLAIMER


# ==================== 全局状态 & 辅助 ====================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def _cur_colors():
    return config.get_colors(st.session_state.dark_mode)


# ==================== 样式 ====================
def inject_css():
    C = _cur_colors()
    _dark = st.session_state.dark_mode
    _sb_bg = C["card_bg"] if _dark else "#ffffff"
    _sb_bd = C["card_border"] if _dark else "#dbe4ee"
    _sb_tx = C["text"] if _dark else "#1e293b"
    _sb_ac = C["accent"] if _dark else "#2B6CB0"
    st.markdown(f"""
    <style>
    .main .block-container {{padding-top: 1.2rem; max-width: 1440px;}}
    h1,h2,h3,h4 {{color: {C['primary']};}}
    .stApp {{background: {C['bg']};
        font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC",
                     "Helvetica Neue", Arial, sans-serif;}}
    /* 侧边栏：渐变背景 + 精准着色 */
    [data-testid="stSidebar"] {{
        background: {"linear-gradient(180deg, #111C32 0%, #1E293B 100%)" if st.session_state.dark_mode
                     else "linear-gradient(180deg, #1E3A5F 0%, #2B6CB0 100%)"};
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3
        {{color: #ffffff !important;}}
    [data-testid="stSidebar"] .stCaptionContainer p {{color: #cfdbe9 !important;}}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] label p {{color: #dfe7f3 !important;}}
    [data-testid="stSidebar"] [role="radiogroup"] label p {{color: #ffffff !important;}}
    [data-testid="stSidebar"] [data-baseweb="select"] > div
        {{background:{_sb_bg}; border-radius:8px; border:1px solid {_sb_bd};}}
    [data-testid="stSidebar"] [data-baseweb="select"] * {{color:{_sb_tx} !important;}}
    [data-testid="stSidebar"] [data-baseweb="select"] div[aria-selected="true"] i
        {{color:{_sb_ac} !important;}}
    /* 指标卡片 */
    .metric-card {{background:{C['card_bg']}; border-radius:10px; padding:12px 14px;
        box-shadow:0 1px 4px rgba(30,58,95,.08); border:1px solid {C['card_border']};}}
    .metric-card .label {{color:{C['text_secondary']}; font-size:12px; white-space:nowrap; margin-bottom:4px;}}
    .metric-card .value {{color:{C['text']}; font-size:20px; font-weight:700;
        white-space:nowrap; line-height:1.2;}}
    .risk-box {{background:{C['up_bg']}; border-left:4px solid {C['up']}; padding:10px 14px;
        border-radius:6px; margin:10px 0; color:{C['text']}; font-size:13px;}}
    .badge {{display:inline-block; background:{C['primary']}; color:#fff;
        padding:3px 12px; border-radius:14px; font-size:13px; margin-left:6px;}}
    /* 完整表格 */
    .dtable {{width:100%; border-collapse:collapse; font-size:13px; background:{C['card_bg']};}}
    .dtable th {{background:{C['primary']}; color:#fff; padding:6px 10px;
        text-align:left; white-space:nowrap;}}
    .dtable td {{border:1px solid {C['card_border']}; padding:6px 10px; vertical-align:top;
        word-break:break-word; text-align:left; color:{C['text']};}}
    .dtable tr:nth-child(even) td {{background:{C['bg']};}}
    {f"""
    /* === 深色模式：正文与原生组件色彩适配（保留涨跌红绿等语义色） === */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {C['primary']} !important;}}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] em,
    [data-testid="stCaptionContainer"] p, .stCaptionContainer p {{color: {C['text']} !important;}}
    .stApp label, .stApp .stSelectbox label, .stApp .stRadio label {{color: {C['text']} !important;}}
    .stApp .st-bb, .stApp .st-at, .stApp .st-aw {{color: {C['text']} !important;}}
    .stApp .st-emotion-cache-1vt4y43, .stApp .st-emotion-cache-15hul6a {{color: {C['text']} !important;}}
    .stApp .stMetric label, .stApp [data-testid="stMetricLabel"] {{color: {C['text_secondary']} !important;}}
    .stApp .stMetric .value, .stApp [data-testid="stMetricValue"] {{color: {C['text']} !important;}}
    .stApp .stDataFrame, .stApp [data-testid="stTable"] {{color: {C['text']} !important;}}
    .stApp input, .stApp textarea, .stApp [data-baseweb="input"] input {{color: {C['text']} !important; background: {C['card_bg']} !important;}}
    .stApp [data-baseweb="select"] > div {{background: {C['card_bg']} !important; border-color: {C['card_border']} !important;}}
    .stApp [data-baseweb="select"] * {{color: {C['text']} !important;}}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{color: {C['text']} !important;}}
    .stApp [data-baseweb="tab"] {{color: {C['text']} !important;}}
    /* 提示框（info/warning/error/success）：深底亮文，语义由图标与边框保留 */
    .stApp [data-testid="stAlert"] {{
        background-color: {C['card_bg']} !important;
        border: 1px solid {C['card_border']} !important;
    }}
    /* 日期选择日历弹窗 / 下拉菜单等弹出层：深底亮字 */
    .stApp [data-baseweb="popover"] {{
        background: {C['card_bg']} !important;
    }}
    .stApp [data-baseweb="calendar"] {{
        background: {C['card_bg']} !important;
    }}
    .stApp [data-baseweb="calendar"] button,
    .stApp [data-baseweb="calendar"] [data-baseweb="calendar-header"] {{
        color: {C['text']} !important;
    }}
    .stApp [data-baseweb="popover"] [data-baseweb="menu"] li {{
        background: {C['card_bg']} !important;
        color: {C['text']} !important;
    }}
    /* 多选已选标签（tag）：深色半透明底 + 亮字 */
    .stApp [data-baseweb="tag"] {{
        background: rgba(255,255,255,0.08) !important;
        color: {C['text']} !important;
    }}
    /* 侧栏导航按钮：深色半透明底 + 亮字；当前项主题色高亮 */
    [data-testid="stSidebar"] [data-testid="stButton"] > button {{
        color: {C['text']} !important;
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {{
        background: {C['primary']} !important;
        color: #0F172A !important;
        border: none !important;
    }}
    """ if st.session_state.dark_mode else ""}
    </style>""", unsafe_allow_html=True)


# ==================== 缓存数据 ====================
@st.cache_data(ttl=config.CACHE_TTL["spot"], show_spinner="正在拉取基金列表...")
def c_fund_universe(fkey):
    return data.get_fund_universe(fkey)


@st.cache_data(ttl=config.CACHE_TTL["list"], show_spinner="正在拉取基金列表...", max_entries=8)
def c_fund_list(fkey):
    """轻量基金名单（仅代码/名称/板块），长缓存供侧边栏选基，避免切模块重拉行情。"""
    return data.get_fund_list(fkey)


@st.cache_data(ttl=config.CACHE_TTL["spot"], show_spinner="正在拉取实时估值...")
def c_otc_estimation():
    return data.get_otc_estimation()


@st.cache_data(ttl=config.CACHE_TTL["normal"], show_spinner=False, max_entries=256)
def c_fund_hist(code, fkey, start, end):
    return data.get_fund_hist(code, fkey, start, end)


@st.cache_data(ttl=config.CACHE_TTL["normal"], show_spinner=False, max_entries=64)
def c_fund_rank(ftype):
    return data.get_open_fund_rank(ftype)


@st.cache_data(ttl=config.CACHE_TTL["long"], show_spinner=False, max_entries=256)
def c_holdings(code):
    return data.get_fund_holdings(str(code))


@st.cache_data(ttl=config.CACHE_TTL["long"], show_spinner="正在穿透持仓行业与风格...", max_entries=256)
def c_lookthrough(code):
    """前十大重仓股的行业/风格穿透（内部对每只股票做元数据缓存）。"""
    return holdings.stock_lookthrough(data.get_fund_holdings(str(code)))


@st.cache_data(ttl=config.CACHE_TTL["normal"], show_spinner=False, max_entries=16)
def c_benchmark(name):
    return data.get_benchmark(name)


@st.cache_data(ttl=config.CACHE_TTL["long"], show_spinner=False, max_entries=64)
def c_fund_pe(code, name):
    """基金跟踪指数的真实 PE-TTM 历史（date, pe），未能解析时返回空。长缓存。"""
    return data.get_fund_pe_history(code, name)


@st.cache_data(ttl=config.CACHE_TTL["normal"], show_spinner="正在计算指标与AI预测...", max_entries=64)
def c_analysis(code, fkey, name, start, end):
    """基金深度分析的重计算（指标/因子校验/AI预测），按基金+区间缓存，加速刷新。"""
    hist = data.get_fund_hist(code, fkey, start, end)
    if hist is None or hist.empty:
        return None
    prices = ind.to_price_series(hist)
    pe_hist = data.get_fund_pe_history(code, name)
    if pe_hist is not None and not pe_hist.empty and "date" in pe_hist.columns:
        pe_hist = pe_hist[(pe_hist["date"] >= pd.Timestamp(start)) &
                          (pe_hist["date"] <= pd.Timestamp(end))]
    metrics = ind.compute_fund_metrics(hist, pe_hist=pe_hist)
    if not metrics:
        return None
    # 因子矩阵只算一次，供 IC 时序/相关性/因子校验/AI 预测复用，消除重复滚动计算
    ff = fac.compute_factor_frame(prices, hist, pe_hist)
    ic_ts, cum_ic = fac.ic_time_series(prices, hist, pe_hist=pe_hist, ff=ff)
    fcorr = fac.factor_correlation_matrix(prices, hist, pe_hist=pe_hist, ff=ff)
    fwarn = fac.high_correlation_warning(fcorr)
    return {
        "hist": hist, "prices": prices, "metrics": metrics,
        "fval": fac.validate_factors(prices, hist, pe_hist=pe_hist, ff=ff),
        "trend": predict.trend_signals(prices),
        "pred": predict.forecast(prices, hist, horizon=20, pe_hist=pe_hist, ff=ff),
        "ic_ts": ic_ts, "cum_ic": cum_ic, "fcorr": fcorr, "fwarn": fwarn,
    }


@st.cache_data(ttl=config.CACHE_TTL["normal"], show_spinner="正在训练多周期预测模型...", max_entries=64)
def c_forecast(code, fkey, name, start, end):
    """AI 走势预测的多周期 LightGBM（4 档），缓存加速。"""
    hist = data.get_fund_hist(code, fkey, start, end)
    if hist is None or hist.empty:
        return None, None
    prices = ind.to_price_series(hist)
    pe_hist = data.get_fund_pe_history(code, name)
    ff = fac.compute_factor_frame(prices, hist, pe_hist)
    mc = predict.multi_horizon_forecast(prices, hist, horizons=(7, 20, 60, 120),
                                        pe_hist=pe_hist, ff=ff)
    trend = predict.trend_signals(prices)
    return mc, trend


@st.cache_data(show_spinner=False, ttl=config.CACHE_TTL["short"])
def c_calibration(code, fkey, name, start, end):
    """概率校准曲线数据（基于默认 20 日 horizon），缓存加速。"""
    hist = data.get_fund_hist(code, fkey, start, end)
    if hist is None or hist.empty:
        return {}
    prices = ind.to_price_series(hist)
    pe_hist = data.get_fund_pe_history(code, name)
    ff = fac.compute_factor_frame(prices, hist, pe_hist)
    return predict.calibration_curve_data(prices, hist, pe_hist, horizon=20, n_bins=10,
                                          ff=ff)


def default_start():
    return (_dt.date.today() - _dt.timedelta(days=3 * 365)).strftime("%Y%m%d")


def default_end():
    return _dt.date.today().strftime("%Y%m%d")


# ==================== 通用组件 ====================
def metric_card(label, value, color=None, suffix="", container=None):
    c = f"color:{color};" if color else ""
    target = container if container is not None else st
    target.markdown(f"""<div class="metric-card"><div class="label">{label}</div>
        <div class="value" style="{c}">{value}{suffix}</div></div>""",
        unsafe_allow_html=True)


def render_metrics_row(metrics, keys_labels, per_row=4):
    for i in range(0, len(keys_labels), per_row):
        group = keys_labels[i:i + per_row]
        cols = st.columns(len(group))
        for col, (k, label) in zip(cols, group):
            v = metrics.get(k, "--")
            if pd.isna(v):
                v = ""
            metric_card(label, v, container=col)


def full_table(df: pd.DataFrame):
    """完整表格：直接渲染全部内容（不截断），长文本自动换行。"""
    html = df.to_html(index=False, na_rep="", border=0, escape=True)
    html = html.replace('class="dataframe"', 'class="dtable"', 1)
    st.markdown(f"<div style='overflow-x:auto;'>{html}</div>", unsafe_allow_html=True)


def _fund_labels(u: pd.DataFrame):
    u = u.dropna(subset=["name"]).copy()
    u["label"] = u["code"].astype(str) + " " + u["name"].astype(str)
    return u


NAV_TREE = [
    ("", [("数据驾驶舱", "📊")]),
    ("单基金分析", [
        ("基金画像", "🔍"),
        ("基金筛选", "⚖️"),
        ("定投测算", "💰"),
        ("策略回测对比", "📈"),
        ("压力测试（单基金）", "🛡️"),
        ("一键量化报告", "📄"),
    ]),
    ("多基金分析", [
        ("基金对比", "🔄"),
        ("组合构建", "🧩"),
        ("大类资产配置", "🌐"),
        ("压力测试（组合）", "🛡️"),
    ]),
]


def sidebar():
    st.sidebar.title("基金量化分析系统")
    st.sidebar.caption("数据源：AKShare 公开行情")
    _d1, _d2 = st.sidebar.columns([3, 1])
    _d1.caption("深色模式")
    dark = _d2.checkbox("深色模式", value=st.session_state.dark_mode,
                         key="dark_toggle", label_visibility="collapsed")
    st.session_state.dark_mode = dark
    st.session_state.setdefault("page", "数据驾驶舱")
    for section, items in NAV_TREE:
        if section:
            st.sidebar.markdown(f"**{section}**")
        for name, emoji in items:
            active = st.session_state["page"] == name
            if st.sidebar.button(f"{emoji} {name}", key=f"nav_{name}",
                                 use_container_width=True,
                                 type="primary" if active else "secondary"):
                st.session_state["page"] = name
                st.rerun()

    st.sidebar.markdown("---")
    st.session_state.style = st.sidebar.selectbox(
        "投资风格", ["保守", "稳健", "进取"], index=1)
    st.session_state.horizon = st.sidebar.selectbox(
        "投资周期", ["短期（1年内）", "中期（1-3年）", "长期（3年以上）"], index=1)

    st.sidebar.markdown("---")
    # 场内/场外 全局筛选
    fc = st.sidebar.selectbox("基金类型", config.FUND_CLASSES,
                              index=config.FUND_CLASSES.index(st.session_state.fund_class))
    st.session_state.fund_class = fc

    # 全局选基金
    fkey = config.FUND_CLASS_KEY[fc]
    u = c_fund_list(fkey)
    if u is None or u.empty or "name" not in u.columns:
        code = st.sidebar.text_input("手动输入基金代码（如 510300）", "510300",
                                     key=f"gb_code_{fkey}")
        st.session_state.code, st.session_state.name = code, code
    else:
        u = _fund_labels(u)
        if fkey == "otc":
            kw = st.sidebar.text_input("搜索基金（名称/代码）", "", key=f"gb_search_{fkey}")
            if kw:
                mask = (u["label"].str.contains(kw, case=False, regex=False)
                        | u["code"].astype(str).str.contains(kw, regex=False))
                found = u[mask].head(300)
                u = found if not found.empty else u.head(300)
            else:
                u = u.head(300)
        labels = u["label"].tolist()
        default = "510300" if fkey == "etf" else None
        idx = 0
        if default:
            for i, c in enumerate(u["code"].astype(str)):
                if c == default:
                    idx = i
                    break
        sel = st.sidebar.selectbox("选择基金", labels, index=idx,
                                   key=f"gb_pick_{fkey}")
        row = u[u["label"] == sel].iloc[0]
        st.session_state.code = str(row["code"])
        st.session_state.name = str(row["name"])

    st.sidebar.markdown("---")
    # 收藏夹（跨会话记忆）
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    fav_label = f"★ 收藏夹（{len(st.session_state.favorites)}）"
    with st.sidebar.expander(fav_label, expanded=False):
        favs = list(st.session_state.favorites)
        if favs:
            for f in favs:
                if st.button(f, key=f"fav_{f}", use_container_width=True):
                    code_candidate = f.split()[0] if f.split() else f
                    st.session_state.code = code_candidate
                    st.session_state.name = f[len(code_candidate)+1:] if len(f) > len(code_candidate) else code_candidate
                    st.rerun()
                    break
            if st.button("清空收藏夹", use_container_width=True):
                st.session_state.favorites.clear()
                st.rerun()
        else:
            st.caption("暂无收藏，在基金画像页可添加。")
    cur_label = f"{st.session_state.code} {st.session_state.name}"
    if st.sidebar.button("收藏当前基金", key="fav_add", use_container_width=True):
        if cur_label not in st.session_state.favorites:
            st.session_state.favorites.append(cur_label)
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"数据更新时间：{_dt.datetime.now():%Y-%m-%d %H:%M}")
    st.sidebar.markdown(
        "<div style='font-size:11px; line-height:1.5; color:#cbd5e1'>仅供研究参考，不构成投资建议</div>"
        f"<div style='font-size:10px; line-height:1.5; color:#9fb3cd; margin-top:4px;'>{RISK}</div>",
        unsafe_allow_html=True)
    return st.session_state["page"]


def cur_fund():
    fkey = config.FUND_CLASS_KEY[st.session_state.fund_class]
    return st.session_state.code, st.session_state.name, fkey


def _universe_for(fkey):
    """返回当前基金类型的基金池（label 已拼好）。"""
    u = c_fund_universe(fkey)
    if u is None or u.empty or "name" not in u.columns:
        return pd.DataFrame()
    return _fund_labels(u)


def _search_narrow(s, key, max_rows=500):
    """场外基金池过大时，用搜索框收窄，返回带 label 的表。"""
    if "label" not in s.columns:
        s = _fund_labels(s)
    kw = st.text_input("搜索基金（名称/代码）", "", key=key)
    if kw:
        mask = (s["label"].str.contains(kw, case=False, regex=False)
                | s["code"].astype(str).str.contains(kw, regex=False))
        s = s[mask]
    return s.head(max_rows)


# ==================== 页面 1：数据驾驶舱 ====================
def page_dashboard():
    C = _cur_colors()
    hcol, bcol = st.columns([4, 1])
    with hcol:
        st.header("数据驾驶舱")
    with bcol:
        st.write("")
        if st.button("刷新实时行情", width="stretch"):
            c_fund_universe.clear()
            c_otc_estimation.clear()
            data.clear_etf_spot_cache()
            data.clear_cache("otc_value_est")
            st.session_state["_refreshed_at"] = _dt.datetime.now()
            st.rerun()

    etf = c_fund_universe("etf")
    otc_est = c_otc_estimation()
    otc_all = c_fund_universe("otc")
    if st.session_state.pop("_refreshed_at", None):
        st.success(f"实时行情已重新拉取（{_dt.datetime.now():%H:%M:%S}）")

    def _tally(df, pct_col="pct_chg"):
        total = int(len(df)) if df is not None else 0
        if df is not None and not df.empty and pct_col in df.columns:
            pc = pd.to_numeric(df[pct_col], errors="coerce")
            up = int((pc > 0).sum())
            down = int((pc < 0).sum())
            flat = int((pc == 0).sum())
            nan_ = int(pc.isna().sum())
            avg = pc.mean()
        else:
            up = down = flat = nan_ = 0
            avg = float("nan")
        return {"total": total, "up": up, "down": down, "flat": flat,
                "nan": nan_, "avg": avg}

    # —— 场内ETF：全量 1621 只 ——
    st.markdown("#### 场内ETF 实时行情")
    if etf is None or etf.empty:
        st.warning("场内ETF 行情拉取失败。")
    else:
        t = _tally(etf)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("场内ETF总数", t["total"])
        with c2:
            metric_card("上涨", t["up"], C["up"])
        with c3:
            metric_card("下跌", t["down"], C["down"])
        with c4:
            _avg = t["avg"]
            metric_card("平均涨跌", f"{_avg:+.2f}%" if pd.notna(_avg) else "--",
                        C["up"] if (pd.notna(_avg) and _avg >= 0)
                        else (C["down"] if pd.notna(_avg) else C["neutral"]))

    # —— 场外基金：全量总数 + 可估算子集 ——
    st.markdown("#### 场外基金 实时估值")
    if otc_est is None or otc_est.empty:
        st.warning("场外基金估值拉取失败。")
    else:
        to = _tally(otc_est)
        # 判断估值数据是否为当日更新（从列名提取估值日期）
        _today = _dt.date.today().strftime("%Y-%m-%d")
        _est_date = ""
        if "est_date" in otc_est.columns and not otc_est.empty:
            _est_date = str(otc_est["est_date"].dropna().iloc[0])[:10] if otc_est["est_date"].dropna().any() else ""
        is_est = _est_date != _today
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("场外基金总数", int(len(otc_all)))
        with c2:
            metric_card("估算上涨" if is_est else "上涨", to["up"], C["up"])
        with c3:
            metric_card("估算下跌" if is_est else "下跌", to["down"], C["down"])
        with c4:
            _oavg = to["avg"]
            metric_card("平均涨跌", f"{_oavg:+.2f}%" if pd.notna(_oavg) else "--",
                        C["up"] if (pd.notna(_oavg) and _oavg >= 0)
                        else (C["down"] if pd.notna(_oavg) else C["neutral"]))
        if is_est:
            st.caption(f"可估算基金数 {to['total']} 只（基于重仓股估算，非当日实时净值）。")

    st.markdown("---")
    st.markdown("#### 按行业/板块分类浏览")
    q1, q2 = st.columns([1, 1])
    with q1:
        q_fc = st.selectbox("基金类型", config.FUND_CLASSES,
                            index=config.FUND_CLASSES.index(st.session_state.fund_class),
                            key="dash_fc")
    q_fkey = config.FUND_CLASS_KEY[q_fc]
    qu = _universe_for(q_fkey)
    if qu.empty:
        st.warning("该类型基金池为空。")
        return

    sectors = sorted(qu["sector"].dropna().unique().tolist())
    with q2:
        sec = st.selectbox("行业/板块", ["全部"] + sectors, key="dash_sec")
    qfiltered = qu if sec == "全部" else qu[qu["sector"] == sec]

    # 实时榜单（按涨跌幅，带排名1-10）
    st.markdown("#### 涨跌幅 TOP 榜")
    top = qfiltered.dropna(subset=["pct_chg"]).sort_values("pct_chg", ascending=False).head(10).copy()
    bot = qfiltered.dropna(subset=["pct_chg"]).sort_values("pct_chg").head(10).copy()
    top.insert(0, "排名", range(1, len(top) + 1))
    bot.insert(0, "排名", range(1, len(bot) + 1))

    def _show_list(df_, title):
        st.markdown(f"**{title}**")
        show = df_[["排名", "code", "name", "price", "pct_chg", "sector"]].rename(columns={
            "code": "代码", "name": "名称", "price": "最新价/估算值", "pct_chg": "涨跌幅%",
            "sector": "板块"})
        st.dataframe(show, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        _show_list(top, "涨幅前10")
    with c2:
        _show_list(bot, "跌幅前10")


# ==================== 页面 2：基金画像（融合深分 + AI 预测 + 因子诊断，Tabs 布局） ====================
def page_profile():
    st.header("基金画像")
    code, name, fkey = cur_fund()
    c1, c2 = st.columns([1, 1])
    with c1:
        start = st.date_input("开始日期", value=_dt.date.today() - _dt.timedelta(days=3 * 365),
                              min_value=_dt.date(2010, 1, 1), max_value=_dt.date.today())
    with c2:
        end = st.date_input("结束日期", value=_dt.date.today(),
                            min_value=start, max_value=_dt.date.today())
    start, end = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    res = c_analysis(code, fkey, name, start, end)
    if res is None:
        st.warning("该基金历史行情为空或样本不足（可能代码有误或区间无数据）。")
        return
    hist = res["hist"]
    prices = res["prices"]
    metrics = res["metrics"].copy()
    fval = res["fval"]
    trend = res["trend"]
    pred = res["pred"]
    _vp = metrics["估值分位"]
    metrics["估值状态"] = ind.valuation_status(_vp / 100) if _vp is not None else "数据不足"
    score = scoring.composite_score(metrics, st.session_state.style)

    tab_names = ["概览", "走势与回撤", "AI 预测", "估值与穿透", "因子诊断"]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # 概览
        if fkey == "otc":
            profile = data.get_fund_profile(code)
            if profile is not None and not profile.empty:
                with st.expander(f"基金档案（{name}）", expanded=False):
                    full_table(profile)

        st.markdown(f"#### {name}（{code}）核心指标")
        render_metrics_row(metrics, [
            ("年化收益", "年化收益%"), ("最大回撤", "最大回撤%"),
            ("夏普比率", "夏普比率"), ("估值分位", "估值分位%"),
            ("年化波动率", "年化波动%"), ("近1年", "近1年%"),
            ("索提诺比率", "索提诺比率"), ("最长回撤天数", "最长回撤天数"),
        ])

        st.markdown(f"#### 综合评分：{score['total']} 分 <span class='badge'>{score['rating']}</span> "
                    f"（风格：{score['style']}）", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(charts.radar_chart(score["breakdown"]), width="stretch")
        with c2:
            st.markdown("**分维度评分明细**")
            pdf = pd.DataFrame([{"维度": k, "权重": f"{d['weight']*100:.0f}%",
                                 "得分": d["score"], "依据": "；".join(d["reasons"])}
                                for k, d in score["breakdown"].items()])
            full_table(pdf)

        st.markdown("#### 滚动收益与风险特征")
        win = st.slider("滚动窗口（交易日）", 30, 120, 60, 10, key="prof_win")
        rr = ind.rolling_return(prices, win)
        rv = ind.rolling_volatility(prices, win)
        pts = ind.rolling_risk_return_points(prices, win, step=max(5, win // 3))
        st.plotly_chart(charts.rolling_line(rr, f"滚动{win}日收益(%)",
                                            _cur_colors()["primary"]), width="stretch")
        st.plotly_chart(charts.rolling_line(rv, f"滚动{win}日年化波动(%)",
                                            _cur_colors()["accent"], fill=True), width="stretch")
        st.plotly_chart(charts.risk_return_scatter(pts), width="stretch")
        st.caption("滚动收益/波动刻画该基金在不同时段收益与波动惯性的变化；"
                   "散点图为滚动子窗口的风险-收益权衡分布，越靠左上越优（高收益低波动）。")

    with tabs[1]:  # 走势与回撤
        if fkey == "otc":
            st.plotly_chart(charts.price_line({name: prices}, "净值走势", normalize=False),
                            width="stretch")
        else:
            st.plotly_chart(charts.candlestick(hist, name), width="stretch")
        st.plotly_chart(charts.drawdown_chart(prices), width="stretch")

        st.markdown("#### 相对基准走势")
        bm = c_benchmark("沪深300")
        bm_p = ind.to_price_series(bm).tail(len(prices))
        prices.index = pd.to_datetime(prices.index)
        bm_p.index = pd.to_datetime(bm_p.index)
        st.plotly_chart(charts.price_line({"基金": prices[-len(bm_p):], "沪深300": bm_p},
                                          "基金 vs 沪深300（归一化）"), width="stretch")

    with tabs[2]:  # AI 预测
        mc, _ = c_forecast(code, fkey, name, start, end)
        if mc:
            st.markdown(f"#### {name} 多周期涨跌概率")
            rows = []
            for h, r in mc.items():
                rows.append({
                    "预测周期": f"{h} 日",
                    "上涨概率": f"{r['prob_up']*100:.1f}%",
                    "下跌概率": f"{r['prob_down']*100:.1f}%",
                    "预期收益区间": f"{r['return_range'][0]:.1f}% ~ {r['return_range'][1]:.1f}%",
                    "置信度": f"{r['confidence']*100:.0f}%",
                    "AUC": r["auc"],
                    "方法": r["method"],
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            cols = st.columns(len(mc))
            for col, (h, r) in zip(cols, mc.items()):
                with col:
                    col.plotly_chart(charts.probability_gauge(r["prob_up"], f"{h}日上涨概率"),
                                     width="stretch")

            st.markdown("#### 拐点与支撑压力")
            t1, t2 = st.columns(2)
            with t1:
                rsi_val = trend.get("rsi")
                signal_val = trend.get("signal", "")
                if rsi_val is not None:
                    st.metric("RSI(14)", rsi_val, delta=signal_val)
                else:
                    st.metric("RSI(14)", "数据不足")
                ma20_val = trend.get("dist_ma20_pct")
                if ma20_val is not None:
                    st.metric("距20日均线", f"{ma20_val:.1f}%")
                else:
                    st.metric("距20日均线", "数据不足")
            with t2:
                sup_val = trend.get("support")
                res_val = trend.get("resistance")
                st.metric("支撑位", f"{sup_val:.4f}" if sup_val is not None else "数据不足")
                st.metric("压力位", f"{res_val:.4f}" if res_val is not None else "数据不足")

            best = max(mc.values(), key=lambda r: r["auc"])
            if best.get("importance"):
                st.markdown("#### 特征重要性 Top8（模型可解释性）")
                imp = best["importance"]
                st.plotly_chart(charts.bar_compare(
                    {k: round(v, 3) for k, v in imp.items()}, "特征重要性"), width="stretch")

            calib = c_calibration(code, fkey, name, start, end)
            if calib:
                st.markdown("#### 概率校准曲线")
                st.plotly_chart(charts.reliability_diagram(calib), width="stretch")
                st.caption("校准曲线检验 LightGBM 输出概率的可靠性：点落在对角线上方表示模型低估、"
                           "下方表示高估；点大小代表该分桶的样本量；n<2 的桶不展示。")
            else:
                st.info("样本外数据不足，无法绘制概率校准曲线（需要 ≥60 条训练样本且 ≥10 条 OOS 校验样本）。")
        else:
            st.warning("AI 预测数据不足，请确保基金有足够的行情数据。")

    with tabs[3]:  # 估值与穿透
        st.markdown("#### 估值偏离监控")
        _vp_pe = metrics.get("估值分位")
        if _vp_pe is not None:
            price_pct = float((prices <= prices.iloc[-1]).mean())
            deviation = float(price_pct - _vp_pe / 100)
            _m1, _m2, _m3 = st.columns(3)
            _m1.metric("价格分位", f"{price_pct:.0%}")
            _m2.metric("估值分位", f"{_vp_pe/100:.0%}")
            _m3.metric("偏离度", f"{deviation:+.0%}")
            if abs(deviation) < 0.15:
                st.info(f"**正常**：价格分位({price_pct:.0%}) ≈ 估值分位({_vp_pe/100:.0%})，估值定价一致")
            elif deviation > 0.15:
                st.error(f"**偏高估**：价格分位({price_pct:.0%}) 显著高于 估值分位({_vp_pe/100:.0%})，"
                         f"偏离 {deviation:.0%}，市场情绪可能过度乐观")
            else:
                st.success(f"**偏低估**：价格分位({price_pct:.0%}) 显著低于 估值分位({_vp_pe/100:.0%})，"
                           f"偏离 {-deviation:.0%}，基本面支撑高于市场定价")
        else:
            st.caption("估值分位数据不足（基金需有可识别的跟踪指数且 PE 历史 >36 条），无法计算估值偏离。")

        holds = c_holdings(code)
        if holds is not None and not holds.empty:
            lt = c_lookthrough(code)
            det = lt.get("details")
            if det is not None and not det.empty:
                st.markdown("#### 底层投向穿透")
                st.markdown(f"前十大合计占净值 **{lt.get('concentration', 0):.1f}%**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.plotly_chart(charts.pie_chart(lt["industry"].to_dict(), "行业分布（占前十）"),
                                    width="stretch")
                with c2:
                    st.plotly_chart(charts.pie_chart(lt["size"].to_dict(), "规模风格（大盘/中盘/小盘）"),
                                    width="stretch")
                with c3:
                    st.plotly_chart(charts.pie_chart(lt["value"].to_dict(), "估值风格（价值/平衡/成长）"),
                                    width="stretch")
                st.caption("行业按重仓股申万一级行业归集；规模风格按总市值（亿）分档（≥800大盘/≥200中盘/其余小盘）；"
                           "估值风格按市盈率（动态）分档（≤20价值/≥40成长/中间平衡），亏损股归入成长。")
                holds_cols = [c for c in ["序号", "股票代码", "股票名称", "占净值比例", "持股数", "持仓市值"]
                              if c in holds.columns]
                st.markdown("##### 前十大重仓股")
                full_table(holds[holds_cols].head(10))
        else:
            st.info("未获取到重仓股数据。")

    with tabs[4]:  # 因子诊断
        st.markdown("#### 因子有效性校验（自动淘汰机制）")
        full_table(fval)

        fval_names = list(fval["因子"]) if fval is not None and not fval.empty else []
        if fval_names:
            st.markdown("#### 因子 IC 时序与累计 IC")
            sel = st.multiselect("选择因子", fval_names, default=fval_names[:5], key="prof_fac")
            ic_frame = res.get("ic_ts")
            cum_ic = res.get("cum_ic")
            if sel and ic_frame is not None and not ic_frame.empty:
                ic_sel = ic_frame[[c for c in sel if c in ic_frame.columns]]
                cum_sel = cum_ic[[c for c in sel if c in cum_ic.columns]]
                st.plotly_chart(charts.factor_ic_chart(ic_sel, "滚动 IC 时序"), width="stretch",
                                key="prof_fac_ic_roll")
                st.plotly_chart(charts.factor_ic_chart(cum_sel, "累计 IC 曲线", cumulative=True),
                                width="stretch", key="prof_fac_ic_cum")
        st.markdown("---")
        st.caption("因子有效性基于滚动 IC 时序统计：IC 均值 >0 且 t 检验显著才标记为可用。"
                   "失败比例 ≥40% 预警、≥60% 已下线。")

        fcorr = res.get("fcorr")
        fwarn = res.get("fwarn")
        if fcorr is not None and not fcorr.empty and fcorr.shape[0] >= 2:
            st.markdown("---")
            st.markdown("#### 因子相关性矩阵（Spearman）")
            if fwarn:
                for a, b, v in fwarn:
                    st.warning(f"高相关因子对：**{a}** ↔ **{b}**（ρ={v}），可能存在冗余")
            st.plotly_chart(charts.corr_heatmap(fcorr, title="因子间 Spearman 相关系数"),
                            width="stretch", key="prof_fac_corr")


# ==================== 页面 4：基金筛选优选 ====================
def page_screener():
    st.header("基金筛选优选")
    fkey = config.FUND_CLASS_KEY[st.session_state.fund_class]

    if fkey == "otc":
        c1, c2 = st.columns([1, 1])
        with c1:
            ftype = st.selectbox("基金类型", config.FUND_TYPES, index=0)
        with c2:
            min_score = st.slider("最低综合评分", 0, 100, 70, 5)
        with st.spinner("拉取开放式基金排行..."):
            rank = c_fund_rank(ftype)
        if rank is None or rank.empty:
            st.warning("数据拉取失败。")
            return
        scored = scoring.score_universe(rank)
        st.markdown(f"共 {len(scored)} 只场外基金，按综合评分排序（现仅基于收益维度；风险评估需进入单基金详查）")
        show = scored[scored["综合评分"] >= min_score].copy()
        cols = [c for c in ["code", "name", "m1", "m3", "m6", "y1", "ytd", "综合评分", "评级"] if c in show.columns]
        show.insert(0, "排名", range(1, len(show) + 1))
        st.dataframe(show[["排名"] + cols].rename(columns={
            "code": "代码", "name": "简称", "m1": "近1月%", "m3": "近3月%",
            "m6": "近6月%", "y1": "近1年%", "ytd": "今年来%"}), width="stretch", hide_index=True)
        st.markdown("#### 收益对比 TOP10")
        top10 = scored.head(10)
        st.plotly_chart(charts.bar_compare(
            dict(zip(top10["name"].astype(str), top10["y1"].fillna(0))), "近1年收益 TOP10 (%)"),
            width="stretch")
    else:
        # 场内ETF：实时行情 + 板块筛选 + 流动性排序
        u = _universe_for("etf")
        if u.empty:
            st.warning("场内ETF行情不可用。")
            return
        sectors = ["全部"] + sorted(u["sector"].dropna().unique().tolist())
        c1, c2 = st.columns([1, 1])
        with c1:
            sec = st.selectbox("行业/板块", sectors, key="scr_sec")
        with c2:
            sort_by = st.selectbox("排序依据", ["涨跌幅%", "成交额", "流通市值"], key="scr_sort")
        f = u if sec == "全部" else u[u["sector"] == sec]
        for c in ["pct_chg", "amount", "流通市值"]:
            if c in f.columns:
                f[c] = pd.to_numeric(f[c], errors="coerce")
        sort_key = {"涨跌幅%": "pct_chg", "成交额": "amount", "流通市值": "流通市值"}[sort_by]
        f = f.sort_values(sort_key, ascending=False).head(50).copy()
        f.insert(0, "排名", range(1, len(f) + 1))
        cols = [c for c in ["code", "name", "price", "pct_chg", "amount", "流通市值", "sector"] if c in f.columns]
        st.markdown(f"共 {len(f)} 只场内ETF 实时行情（按 {sort_by} 排序前50），"
                    f"详细指标请进入「基金画像」查看。")
        st.dataframe(f[["排名"] + cols].rename(columns={
            "code": "代码", "name": "名称", "price": "最新价", "pct_chg": "涨跌幅%",
            "amount": "成交额", "sector": "板块"}), width="stretch", hide_index=True)


# ==================== 页面 5：基金对比 ====================
def page_compare():
    st.header("基金横向对比")
    fkey = config.FUND_CLASS_KEY[st.session_state.fund_class]
    s = _universe_for(fkey)
    if s.empty:
        st.warning("数据不可用")
        return
    # 场外基金池过大，先用搜索收窄候选（即时过滤）
    if fkey == "otc":
        s = _search_narrow(s, "cmp_search", max_rows=300)

    with st.form("cmp_form"):
        options = s["label"].tolist()
        default = options[:2] if len(options) >= 2 else options
        picks = st.multiselect("选择对比基金（2~6 只，可搜索）", options, default=default,
                               max_selections=6, key="cmp_picks")
        t1, t2 = st.columns(2)
        with t1:
            start_date = st.date_input("开始时间", value=_dt.date.today() - _dt.timedelta(days=3 * 365),
                                       min_value=_dt.date(2010, 1, 1), max_value=_dt.date.today())
        with t2:
            end_date = st.date_input("结束时间", value=_dt.date.today(),
                                     min_value=start_date, max_value=_dt.date.today())
        submitted = st.form_submit_button("开始对比", type="primary", use_container_width=True)

    if submitted:
        st.session_state["cmp_done"] = True
    if not st.session_state.get("cmp_done", False):
        st.info("选择对比基金并设置日期后，点击「开始对比」。")
        return

    if len(picks) < 2:
        st.warning("请至少选择 2 只基金进行对比。")
        return
    if start_date >= end_date:
        st.warning("开始时间需早于结束时间")
        return
    start, end = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")

    with st.spinner("拉取各基金行情并计算对比指标..."):
        metrics_map, prices_map, score_map, rating_map = {}, {}, {}, {}
        for lbl in picks:
            hit = s[s["label"] == lbl]
            if hit.empty:
                # 基金类型切换后，已选基金可能不在当前池中，安全跳过
                continue
            c = hit["code"].iloc[0]
            nm = hit["name"].iloc[0]
            h = c_fund_hist(c, fkey, start, end)
            if h is None or h.empty:
                continue
            prices = ind.to_price_series(h)
            pe_hist = c_fund_pe(c, nm)
            m = ind.compute_fund_metrics(h, pe_hist=pe_hist)
            if not m:
                continue
            vp = m["估值分位"]
            m["估值状态"] = ind.valuation_status(vp / 100) if vp is not None else "数据不足"
            sc = scoring.composite_score(m, st.session_state.style)
            metrics_map[nm] = m
            prices_map[nm] = prices
            score_map[nm] = sc["total"]
            rating_map[nm] = sc["rating"]

    if len(metrics_map) < 2:
        st.warning("有效成分不足（部分基金区间内无行情），无法对比。")
        return
    names = list(metrics_map.keys())

    # 指标横向对比表（行为指标、列为基金）
    st.markdown("#### 核心指标横向对比")
    rows = []
    for key, label in [("年化收益", "年化收益(%)"), ("近1年", "近1年(%)"),
                       ("最大回撤", "最大回撤(%)"), ("年化波动率", "年化波动率(%)"),
                       ("夏普比率", "夏普比率"), ("卡玛比率", "卡玛比率"),
                       ("估值分位", "估值分位(%)")]:
        row = {"指标": label}
        for nm in names:
            v = metrics_map[nm].get(key)
            row[nm] = f"{v:.2f}" if isinstance(v, (int, float)) else "--"
        rows.append(row)
    full_table(pd.DataFrame(rows))

    # 净值与回撤改为整行堆叠展示，避免并排时图例/时间轴拥挤
    st.plotly_chart(charts.price_line(prices_map, "归一化净值对比"), width="stretch")
    st.plotly_chart(charts.drawdown_compare(prices_map, "回撤对比"), width="stretch")

    st.markdown("#### 四维评分雷达对比")
    breakdowns = {nm: scoring.composite_score(metrics_map[nm], st.session_state.style)["breakdown"]
                  for nm in names}
    st.plotly_chart(charts.radar_overlay(breakdowns), width="stretch")

    st.markdown("#### 综合评分排名")
    rank_df = pd.DataFrame([{"基金": nm, "综合评分": score_map[nm], "评级": rating_map[nm]}
                            for nm in names]).sort_values("综合评分", ascending=False)
    full_table(rank_df)
    st.plotly_chart(charts.bar_compare({nm: score_map[nm] for nm in rank_df["基金"]},
                                       "综合评分对比（百分制）"), width="stretch")
    best = rank_df.iloc[0]["基金"]
    st.success(f"在当前风格「{st.session_state.style}」下，综合评分最高：**{best}**"
               f"（{rank_df.iloc[0]['评级']}，{rank_df.iloc[0]['综合评分']} 分）")

    compare_html = report.build_compare_report_html(
        names, metrics_map, {nm: score_map[nm] for nm in names},
        {nm: rating_map[nm] for nm in names},
        breakdowns=breakdowns)
    st.download_button("下载横向对比报告（HTML）", data=compare_html,
                       file_name=f"fund_compare_{pd.Timestamp.now():%Y%m%d}.html")


# ==================== 页面 6：组合优化 ====================
def _invalidate_pf():
    """组合构建页开关/成分变化时作废已缓存结果，避免显示过期数据。"""
    st.session_state.pop("pf_done", None)
    st.session_state.pop("pf_cache", None)
    st.session_state.pop("pf_walk_summary", None)


def page_portfolio():
    st.header("投资组合量化优化")
    st.caption("马科维茨均值-方差 / 风险预算 / 目标收益等多种优化方法")
    fkey = config.FUND_CLASS_KEY[st.session_state.fund_class]
    s = _universe_for(fkey)
    if s.empty:
        st.warning("数据不可用")
        return
    # 场外基金池过大时，先搜索收窄候选（即时过滤）
    if fkey == "otc":
        s = _search_narrow(s, "pf_search", max_rows=300)

    # 选择成分基金（多选，选好后随下方表单一并提交）
    options = s["label"].tolist()
    default = options[:3] if len(options) >= 3 else options
    picks = st.multiselect("选择成分基金（2~10 只，可搜索）", options, default=default,
                           max_selections=10, key="pf_picks", on_change=_invalidate_pf)

    # 优化目标 + 开关（表单外，切换即时生效并作废旧结果）
    mode = st.radio("优化目标", ["最小方差", "最大夏普", "目标收益", "风险预算"],
                    horizontal=True, on_change=_invalidate_pf)
    use_per_asset = st.checkbox("启用逐资产权重边界（覆盖全局上限）", value=False,
                                on_change=_invalidate_pf)
    use_sector = st.checkbox("启用行业/板块集中度上限", value=False, on_change=_invalidate_pf)

    # 优化参数（一次性设置后提交）
    with st.form("pf_form"):
        max_w = st.slider("单资产权重上限（1.0 = 不限制）", 0.2, 1.0, 1.0, 0.05)

        target_ret = 0.1
        if mode == "目标收益":
            target_ret = st.slider("目标年化收益", 0.0, 0.3, 0.1, 0.01)

        bounds = None
        if use_per_asset:
            per_bounds = []
            for i, lbl in enumerate(picks):
                b1, b2 = st.columns(2)
                with b1:
                    lo = st.number_input(f"{lbl} 最小", 0.0, 1.0, 0.0, 0.05,
                                         key=f"pf_lo_{i}", format="%.2f")
                with b2:
                    hi = st.number_input(f"{lbl} 最大", 0.0, 1.0, max_w, 0.05,
                                         key=f"pf_hi_{i}", format="%.2f")
                per_bounds.append((lo, hi))
            bounds = per_bounds

        sector_limit = None
        if use_sector:
            sector_limit = st.slider("单一行业/板块权重上限", 0.2, 1.0, 0.5, 0.05)

        budget = None
        if mode == "风险预算":
            st.caption("各资产目标风险贡献占比（%），提交时自动归一化；默认各资产等额。")
            budget_pct = []
            n_pick = max(len(picks), 1)
            for i, lbl in enumerate(picks):
                budget_pct.append(st.number_input(f"{lbl} 风险预算(%)", 0.0, 100.0,
                                                 round(100.0 / n_pick, 1), 1.0,
                                                 key=f"pf_budget_{i}", format="%.1f"))
            budget = [p / 100.0 for p in budget_pct]

        submitted = st.form_submit_button("开始优化", type="primary", use_container_width=True)

    if submitted:
        st.session_state["pf_done"] = True
    if not st.session_state.get("pf_done", False):
        st.info("选择成分基金并设置优化参数后，点击「开始优化」。")
        return

    if len(picks) < 2:
        st.warning("请至少选择 2 只基金构建组合。")
        return

    # 缓存主优化结果：后续点击「样本外回测」等按钮触发 rerun 时命中缓存，避免整页重算灰屏
    cache_key = (
        fkey, mode, tuple(picks), round(max_w, 4),
        round(target_ret, 4) if mode == "目标收益" else None,
        tuple((round(lo, 4), round(hi, 4)) for lo, hi in (bounds or [])),
        None if sector_limit is None else round(sector_limit, 4),
        None if budget is None else tuple(round(b, 4) for b in budget),
        default_start(), default_end(),
    )
    pf_cache = st.session_state.get("pf_cache")
    if pf_cache is not None and pf_cache["key"] == cache_key:
        prices_dict, ret, opt = pf_cache["prices_dict"], pf_cache["ret"], pf_cache["opt"]
    else:
        st.session_state.pop("pf_walk_summary", None)  # 参数/成分变了，作废旧样本外结果
        with st.spinner("拉取成分行情并优化..."):
            prices_dict = {}
            for lbl in picks:
                c = s[s["label"] == lbl]["code"].iloc[0]
                nm = s[s["label"] == lbl]["name"].iloc[0]
                h = c_fund_hist(c, fkey, default_start(), default_end())
                if not h.empty:
                    prices_dict[nm] = ind.to_price_series(h)
            if len(prices_dict) < 2:
                st.warning("有效成分不足")
                return
            ret = pd.DataFrame({k: v.pct_change() for k, v in prices_dict.items()}).dropna()
            mode_key = {"最大夏普": "max_sharpe", "最小方差": "min_variance",
                        "目标收益": "target_return", "风险预算": "risk_budgeting"}[mode]
            opt = pf.optimize_portfolio(ret, mode=mode_key,
                                        target_return=(target_ret if mode == "目标收益" else None),
                                        max_weight=max_w, bounds=bounds,
                                        sector_limit=sector_limit,
                                        budget=(budget if mode == "风险预算" else None),
                                        return_frontier=True)
        st.session_state["pf_cache"] = {"key": cache_key, "prices_dict": prices_dict,
                                        "ret": ret, "opt": opt}

    if opt.get("note"):
        st.warning(opt["note"])

    st.markdown("#### 最优权重")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(charts.pie_chart(opt["weights"]), width="stretch")
    with c2:
        st.markdown("**组合绩效指标**")
        render_metrics_row(opt["stats"], [
            ("年化收益", "年化收益%"), ("最大回撤", "最大回撤%"),
            ("夏普比率", "夏普比率"), ("年化波动率", "年化波动%"),
            ("VaR95", "VaR95%"), ("CVaR95", "CVaR95%"),
        ])
        wdf = pd.DataFrame([{"资产": k, "权重": f"{v*100:.1f}%" if v >= 0.00005 else "<0.01%"}
                            for k, v in opt["weights"].items()])
        full_table(wdf)

    if opt.get("frontier") is not None and len(opt["frontier"]):
        st.plotly_chart(charts.frontier_chart(opt["frontier"], opt["stats"]),
                        width="stretch")

    st.markdown("#### 风险结构与相关性")
    rc_df = pf.risk_contribution(opt["raw_weights"], ret)
    st.plotly_chart(charts.risk_contribution_bar(rc_df), width="stretch")
    st.plotly_chart(charts.corr_heatmap(ret.corr()), width="stretch")
    st.caption("风险贡献占比合计 100%；占比显著高于权重的资产为组合主要风险来源，"
               "相关性接近 1 的成分无法通过分散化降低风险。")

    st.markdown("#### 组合回测净值")
    nav = pf.composite_nav(opt["weights"], prices_dict)
    st.plotly_chart(charts.price_line({"组合": nav}, "组合净值（起点=1）", normalize=False),
                    width="stretch")

    # ---- 优化方法横向回测对比（滚动样本外） ----
    st.markdown("#### 优化方法横向回测对比（样本外）")
    st.caption("滚动训练窗口、逐步外推，评估各优化方法的样本外表现，避免仅看一次全样本"
               "最优权重带来的过拟合。")
    f1, f2 = st.columns(2)
    with f1:
        train_win = int(st.select_slider("训练窗口（交易日）", options=[60, 126, 252], value=126))
    with f2:
        step = int(st.select_slider("调仓步长（交易日）", options=[5, 10, 20, 60], value=20))
    if st.button("运行样本外回测对比", key="pf_walk"):
        with st.spinner("滚动样本外回测中..."):
            wf = pf.walk_forward_backtest(
                ret, methods=["min_variance", "max_sharpe", "risk_budgeting", "equal_weight"],
                train_window=train_win, step=step, max_weight=max_w,
                sector_limit=sector_limit)
        if not wf:
            st.session_state.pop("pf_walk_summary", None)
            st.warning("历史数据不足以进行滚动样本外回测（需超过训练窗口+调仓步长）。")
        else:
            summary = pf.walk_forward_summary(wf)
            st.session_state["pf_walk_summary"] = summary
            st.markdown("**样本外绩效对比**（年化收益/回撤/波动/VaR/CVaR 单位：%）")
            full_table(summary)
            navs = {k: v["nav"] for k, v in wf.items()}
            st.plotly_chart(charts.price_line(navs, "各优化方法样本外净值（起点=1）",
                                             normalize=False), width="stretch")
            best = summary.iloc[0]["方法"]
            st.success(f"样本外表现最优（按夏普比率）：**{best}**")

    walk_summary = st.session_state.get("pf_walk_summary")
    pf_html = report.build_portfolio_report_html(
        opt, ret, prices_dict, mode,
        {"最大夏普": "最大夏普比率", "最小方差": "最小方差",
         "目标收益": "目标收益优化", "风险预算": "风险预算"}.get(mode, mode),
        walk_forward=walk_summary)
    st.download_button("下载组合优化报告（HTML）", data=pf_html,
                       file_name=f"portfolio_{pd.Timestamp.now():%Y%m%d}.html")


# ==================== 页面 7：大类资产配置 ====================
def page_allocation():
    st.header("大类资产配置")
    st.caption("基于 股票/债券/黄金/现金 四类代表性 ETF（沪深300 / 国债 / 黄金 / 货币ETF）真实行情，"
               "提供跨资产配置与周期性再平衡回测。")

    with st.form("alloc_form"):
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("开始日期", value=_dt.date(2015, 1, 1),
                                  min_value=_dt.date(2010, 1, 1), max_value=_dt.date.today())
        with c2:
            end = st.date_input("结束日期", value=_dt.date.today(),
                                min_value=start, max_value=_dt.date.today())

        m1, m2 = st.columns(2)
        with m1:
            method = st.selectbox("配置方法", allocation.ALLOCATION_METHODS, index=3)
        with m2:
            target_vol = st.slider("目标波动率(年化 · 仅「目标波动」方法生效)", 0.03, 0.25, 0.08, 0.01)

        submitted = st.form_submit_button("开始配置", type="primary", use_container_width=True)

    if submitted:
        st.session_state["alloc_done"] = True
    if not st.session_state.get("alloc_done", False):
        st.info("设置好日期、配置方法与目标波动率后，点击「开始配置」。")
        return

    if start >= end:
        st.warning("开始日期需早于结束日期")
        return
    start, end = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    # 缓存配置计算结果：后续点击「再平衡回测」等按钮触发 rerun 时命中缓存，避免整页重算灰屏
    cache_key = (start, end, method, round(target_vol, 4))
    alloc_cache = st.session_state.get("alloc_cache")
    if alloc_cache is not None and alloc_cache["key"] == cache_key:
        prices_dict = alloc_cache["prices_dict"]
        ret = alloc_cache["ret"]
        weights = alloc_cache["weights"]
        stats = alloc_cache["stats"]
        summary_df = alloc_cache["summary_df"]
    else:
        st.session_state.pop("alloc_rb", None)  # 参数变了，作废旧再平衡结果
        with st.spinner("拉取大类资产行情..."):
            prices_dict, ret = allocation.asset_class_returns(start, end)
        if ret.empty or ret.shape[1] < 2:
            st.warning("大类资产行情数据不足，请扩大区间或稍后重试。")
            return
        weights, stats = allocation.method_perf(ret, method, target_vol)
        summary_df = allocation.allocation_summary(ret, target_vol=target_vol)
        st.session_state["alloc_cache"] = {"key": cache_key, "prices_dict": prices_dict,
                                           "ret": ret, "weights": weights,
                                           "stats": stats, "summary_df": summary_df}

    st.markdown("#### 各类资产历史走势（归一化 = 1）")
    st.plotly_chart(charts.price_line(prices_dict, "大类资产归一化走势"), width="stretch")

    st.markdown("#### 目标配置权重")
    w1, w2 = st.columns([1, 1])
    with w1:
        st.plotly_chart(charts.pie_chart(weights), width="stretch")
    with w2:
        st.markdown("**配置绩效（全样本）**")
        render_metrics_row(stats, [
            ("年化收益", "年化收益%"), ("最大回撤", "最大回撤%"),
            ("夏普比率", "夏普比率"), ("年化波动率", "年化波动%"),
            ("VaR95", "VaR95%"), ("CVaR95", "CVaR95%"),
        ])
        wdf = pd.DataFrame([{"资产": k, "权重": f"{v*100:.1f}%"} for k, v in weights.items()])
        full_table(wdf)

    st.markdown("#### 各配置方法绩效对比（全样本）")
    full_table(summary_df)

    st.markdown("#### 再平衡回测（vs 买入持有）")
    freq_map = {"月度": 21, "季度": 63, "年度": 252}
    f1, f2 = st.columns(2)
    with f1:
        freq = st.selectbox("再平衡频率", list(freq_map.keys()), index=1)
    with f2:
        lookback = int(st.select_slider("权重观察窗口(交易日)", options=[126, 252, 504], value=252))
    if st.button("运行再平衡回测", key="alloc_rebal"):
        with st.spinner("回测中..."):
            rb = allocation.rebalance_backtest(ret, method, target_vol,
                                               rebalance_days=freq_map[freq],
                                               lookback=lookback)
        if not rb:
            st.session_state.pop("alloc_rb", None)
            st.warning("历史数据不足以进行再平衡回测（需超过观察窗口+再平衡周期）。")
        else:
            navs = {k: v["nav"] for k, v in rb.items()}
            st.plotly_chart(charts.price_line(navs, "再平衡 vs 买入持有（起点=1）", normalize=False),
                            width="stretch")
            rb_df = pd.DataFrame([{"策略": k, **v["stats"]} for k, v in rb.items()])
            st.session_state["alloc_rb"] = rb_df
            full_table(rb_df)
            rb_s, bh_s = rb["再平衡"]["stats"], rb["买入持有"]["stats"]
            if rb_s["夏普比率"] >= bh_s["夏普比率"]:
                st.success(f"再平衡夏普 {rb_s['夏普比率']:.2f} ≥ 买入持有 {bh_s['夏普比率']:.2f}，"
                           f"定期再平衡带来风险收益改善。")
            else:
                st.info(f"再平衡夏普 {rb_s['夏普比率']:.2f} < 买入持有 {bh_s['夏普比率']:.2f}，"
                        f"受趋势/交易成本影响，本次区间再平衡未跑赢。")

    st.caption(config.RISK_DISCLAIMER)

    alloc_html = report.build_allocation_report_html(method, weights, stats,
                                                     rebalance=st.session_state.get("alloc_rb"))
    st.download_button("下载大类资产配置报告（HTML）", data=alloc_html,
                       file_name=f"allocation_{pd.Timestamp.now():%Y%m%d}.html")


# ==================== 页面 8：定投测算 ====================
def page_dca():
    st.header("基金定投测算")
    st.caption("支持固定金额、估值不定额、首笔初投等多种定投策略的历史回测")
    code, name, fkey = cur_fund()

    with st.form("dca_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            freq = st.selectbox("定投频率", ["周", "双周", "月"], index=2)
        with c2:
            amount = st.number_input("每期金额(元)", 100, 100000, 1000, 100)
        with c3:
            mode = st.selectbox("定投模式", ["固定", "估值不定额"], index=0)
        with c4:
            init_amt = st.number_input("初投金额(元)", 0, 1000000, 0, 100,
                                       help="起投时一次性投入的金额，0 表示无首笔投入")

        t1, t2 = st.columns(2)
        with t1:
            start_date = st.date_input("起投时间", value=_dt.date.today() - _dt.timedelta(days=5 * 365),
                                       min_value=_dt.date(2010, 1, 1),
                                       max_value=_dt.date.today())
        with t2:
            end_date = st.date_input("结束时间", value=_dt.date.today(),
                                     min_value=start_date, max_value=_dt.date.today())
        submitted = st.form_submit_button("开始测算", type="primary", use_container_width=True)

    if submitted:
        st.session_state["dca_done"] = True
    if not st.session_state.get("dca_done", False):
        st.info("设置好定投参数后，点击「开始测算」。")
        return

    if start_date >= end_date:
        st.warning("起投时间需早于结束时间")
        return
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    hist = c_fund_hist(code, fkey, "20100101", end)
    if hist is None or hist.empty:
        st.warning("无数据")
        return
    prices = ind.to_price_series(hist)
    pe_hist = c_fund_pe(code, name)

    with st.spinner("定投回测计算中..."):
        res = dca.dca_backtest(prices, start=start, end=end, freq=freq,
                               amount=amount, mode=mode, initial_amount=float(init_amt),
                               pe_hist=pe_hist)
        stop = dca.dca_stop_profit_backtest(prices, start=start, end=end,
                                            freq=freq, amount=amount, pe_hist=pe_hist)
        timing = dca.dca_timing_advice(prices, pe_hist=pe_hist)

    if not res:
        st.warning("数据不足")
        return

    st.markdown(f"#### {name} 定投结果（{mode} · {freq} · 起投 {start_date}）")
    render_metrics_row(res, [
        ("总投入", "总投入(元)"), ("期末资产", "期末资产(元)"),
        ("收益率", "总收益率%"), ("定投IRR", "年化IRR%"),
        ("最大回撤", "最大回撤%"), ("定投期数", "定投期数"),
    ])

    st.plotly_chart(charts.smile_chart(res["微笑曲线"]), width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**时点优选建议**")
        st.info(f"估值分位 {timing['估值分位']}% | 波动率 {timing['波动率']}%\n\n{timing['建议']}")
    with c2:
        st.markdown("**止盈策略历史模拟对比**")
        full_table(stop["止盈对比"])

    st.caption("定投失效提示：长期单边下跌且无反弹的市场中，定投可能持续亏损，"
               "需结合估值与基本面判断，不适用于趋势性崩盘资产。")


# ==================== 页面 9：策略回测对比 ====================
def page_backtest():
    st.header("历史回测与策略对比")
    st.caption("双均线 / 动量 / 定投 / 买入持有 多策略横向对比，计入交易成本")
    code, name, fkey = cur_fund()

    with st.form("bt_form"):
        c1, c2 = st.columns([1, 1])
        with c1:
            start = st.date_input("开始日期", value=_dt.date.today() - _dt.timedelta(days=5 * 365),
                                  min_value=_dt.date(2010, 1, 1), max_value=_dt.date.today())
        with c2:
            end = st.date_input("结束日期", value=_dt.date.today(),
                                min_value=start, max_value=_dt.date.today())

        st.markdown("#### 交易成本（择时策略计入 · 单位：万分之几）")
        t1, t2 = st.columns([1, 1])
        with t1:
            fee = st.number_input("单边手续费（万分之几）", 0.0, 50.0, 2.5, 0.5,
                                  format="%.1f", help="如万 2.5 手续费；万分之 1 = 0.01%")
        with t2:
            slip = st.number_input("单边滑点成本（万分之几）", 0.0, 50.0, 5.0, 0.5,
                                   format="%.1f", help="买卖价差 + 冲击成本，如万 5")

        st.markdown("#### 策略参数")
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            ma_fast = st.number_input("双均线·短周期", 3, 250, 20, 1)
        with p2:
            ma_slow = st.number_input("双均线·长周期", 5, 500, 60, 1)
        with p3:
            mom_window = st.number_input("动量·回顾窗口(日)", 5, 250, 60, 1)
        with p4:
            dca_freq = st.selectbox("定投频率", ["周", "双周", "月"], index=2)
        with p5:
            dca_amount = st.number_input("定投每期金额(元)", 100, 100000, 1000, 100)
        submitted = st.form_submit_button("开始回测", type="primary", use_container_width=True)

    if submitted:
        st.session_state["bt_done"] = True
    if not st.session_state.get("bt_done", False):
        st.info("设置好回测参数后，点击「开始回测」。")
        return

    start, end = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    cost_rate = (fee + slip) / 10000.0

    if ma_fast >= ma_slow:
        st.warning("双均线短周期需小于长周期。")
        return

    hist = c_fund_hist(code, fkey, start, end)
    if hist is None or hist.empty:
        st.warning("该区间无历史行情数据，请更换区间或基金。")
        return
    prices = ind.to_price_series(hist)
    pe_hist = c_fund_pe(code, name)

    with st.spinner("正在并行回测五套策略..."):
        results, summary = backtest.compare_strategies(
            prices, pe_hist, ma_fast=ma_fast, ma_slow=ma_slow, mom_window=mom_window,
            cost_rate=cost_rate, dca_freq=dca_freq, dca_amount=float(dca_amount),
            start=start, end=end)

    if not results:
        st.warning("该区间样本不足，无法完成回测，请扩大时间范围。")
        return

    st.markdown(f"#### {name}（{code}）五策略回测对比　（单边合计成本 万分之 {(fee + slip):.1f}）")

    show = summary.copy()

    def _pct(v):
        return f"{v * 100:.2f}%" if pd.notna(v) else "--"

    def _num(v):
        return f"{v:.2f}" if pd.notna(v) else "--"

    show["总收益"] = show["总收益"].map(_pct)
    show["年化收益"] = show["年化收益"].map(_pct)
    show["年化波动"] = show["年化波动"].map(_pct)
    show["最大回撤"] = show["最大回撤"].map(_pct)
    show["夏普比率"] = show["夏普比率"].map(_num)
    show["卡玛比率"] = show["卡玛比率"].map(_num)
    show["交易次数"] = show["交易次数"].map(lambda v: f"{int(v)}" if pd.notna(v) else "--")
    st.dataframe(show, width="stretch", hide_index=True)

    navs = {r["name"]: r["nav"] for r in results.values() if len(r["nav"]) > 1}
    st.markdown("#### 各策略净值走势（起点 = 1）")
    st.plotly_chart(charts.price_line(navs, "各策略净值对比", normalize=False), width="stretch")

    st.markdown("#### 各策略回撤对比")
    st.plotly_chart(charts.drawdown_compare(navs), width="stretch")

    best_sharpe = summary.dropna(subset=["夏普比率"])
    if not best_sharpe.empty:
        best = best_sharpe.sort_values("夏普比率", ascending=False).iloc[0]
        st.markdown("#### 回测结论")
        st.info(f"夏普比率最高的策略为 **{best['策略']}**（夏普 {best['夏普比率']:.2f}）。"
                f"历史回测存在幸存者偏差，不代表未来收益；"
                f"趋势/动量策略在震荡市中可能因频繁换仓被交易成本侵蚀。")
    st.caption(config.RISK_DISCLAIMER)


# ==================== 页面 10：压力测试 ====================
def page_stress(preset_mode=None):
    st.header("极端行情压力测试")
    st.caption("历史情景回放 + 蒙特卡洛模拟，评估单基金或组合在极端条件下的尾部风险")
    code, name, fkey = cur_fund()
    if preset_mode:
        mode = preset_mode
    else:
        mode = st.radio("测试对象", ["单基金", "组合"], horizontal=True)
    if mode == "单基金":
        hist = c_fund_hist(code, fkey, "20150101", default_end())
        if hist is None or hist.empty:
            st.warning("无数据"); return
        prices = ind.to_price_series(hist)
        res, navs = stress.stress_test_fund_with_paths(prices)
        obj_name = f"{name}（{code}）"
    else:
        s = _universe_for(fkey)
        if s.empty:
            st.warning("数据不可用"); return
        if fkey == "otc":
            s = _search_narrow(s, "stress_search")
        if fkey == "etf":
            default_codes = ["510300", "510500", "159915"]
        else:
            default_codes = s["code"].astype(str).head(3).tolist()
        labels = s[s["code"].astype(str).isin(default_codes)]["label"].tolist()
        picks = st.multiselect("选择成分基金（可搜索）", s["label"].tolist(),
                               default=labels or s["label"].tolist()[:3], key="stress_picks")
        if len(picks) < 2:
            st.info("请至少选 2 只"); return
        # 预取成分行情（缓存）用于渲染权重输入
        prices_dict = {}
        for lbl in picks:
            c = s[s["label"] == lbl]["code"].iloc[0]
            nm = s[s["label"] == lbl]["name"].iloc[0]
            h = c_fund_hist(c, fkey, "20150101", default_end())
            if not h.empty:
                prices_dict[nm] = ind.to_price_series(h)
        if len(prices_dict) < 2:
            st.warning("有效成分不足"); return
        with st.form("stress_port_form"):
            st.markdown("**设置各基金比例（自动归一化为 100%）**")
            raw_w = {}
            wcols = st.columns(min(len(prices_dict), 4))
            for i, (nm, _) in enumerate(prices_dict.items()):
                with wcols[i % len(wcols)]:
                    raw_w[nm] = st.number_input(f"{nm}", 0.0, 100.0,
                                                round(100.0 / len(prices_dict), 2), 1.0,
                                                key=f"stress_w_{i}", format="%.1f")
            submitted = st.form_submit_button("开始压力测试", type="primary", use_container_width=True)
        if submitted:
            st.session_state["stress_port_done"] = True
        if not st.session_state.get("stress_port_done", False):
            st.info("设置各基金比例后，点击「开始压力测试」。")
            return
        total_w = sum(raw_w.values())
        weights = ({k: v / total_w for k, v in raw_w.items()} if total_w > 0
                   else {k: 1.0 / len(raw_w) for k in raw_w})
        st.caption("实际归一化权重：" + "，".join(
            f"{k} {v*100:.1f}%" for k, v in weights.items()))
        res, navs = stress.stress_test_portfolio_with_paths(prices_dict, weights)
        obj_name = "当前组合"

    st.markdown(f"#### {obj_name} 压力测试（历史回放 + 蒙特卡洛）")
    sdf = pd.DataFrame([
        {"情景": k, **{kk: vv for kk, vv in v.items() if kk != "说明"}, "说明": v.get("说明", "")}
        for k, v in res.items()
    ]).rename(columns={"修复周期(交易日)": "修复周期"})
    full_table(sdf)

    st.plotly_chart(charts.stress_bar(res), width="stretch")
    if navs and any(len(v) > 1 for v in navs.values()):
        st.plotly_chart(charts.stress_drawdown_overlay(navs), width="stretch")
    else:
        st.info("暂无足够数据绘制回撤路径曲线")
    st.markdown("#### 结论")
    st.warning(stress.stress_summary(res))

    st.markdown("#### 尾部风险（蒙特卡洛 VaR / CVaR）")
    if mode == "单基金":
        tail_rows = [
            {"指标": "历史 VaR95", "数值(%)": round(ind.var_historical(prices) * 100, 2)},
            {"指标": "历史 CVaR95", "数值(%)": round(ind.cvar_historical(prices) * 100, 2)},
            {"指标": "蒙特卡洛 VaR95", "数值(%)": round(ind.var_monte_carlo(prices) * 100, 2)},
            {"指标": "蒙特卡洛 CVaR95", "数值(%)": round(ind.cvar_monte_carlo(prices) * 100, 2)},
        ]
        full_table(pd.DataFrame(tail_rows))
        st.caption("蒙特卡洛 VaR/CVaR 基于几何布朗运动 1 日模拟（20000 条路径、95% 置信度）。")
    else:
        ret = pd.DataFrame({k: v.pct_change() for k, v in prices_dict.items()}).dropna()
        tr = pf.portfolio_tail_risk(ret, weights)
        tail_rows = [
            {"指标": "历史 VaR95", "数值": f"{tr['历史VaR95']}%"},
            {"指标": "历史 CVaR95", "数值": f"{tr['历史CVaR95']}%"},
            {"指标": "蒙特卡洛 VaR95", "数值": f"{tr['蒙特卡洛VaR95']}%"},
            {"指标": "蒙特卡洛 CVaR95", "数值": f"{tr['蒙特卡洛CVaR95']}%"},
            {"指标": "平均相关性", "数值": tr["平均相关性"]},
            {"指标": "分散化收益", "数值": f"{tr['分散化收益']}%"},
        ]
        full_table(pd.DataFrame(tail_rows))
        st.caption(tr["说明"] + "。分散化收益为正表示相关性令组合尾部损失小于各资产独立相加。")

    if mode == "组合":
        stress_html = report.build_stress_report_html(obj_name, res, navs=navs if navs else None)
        st.download_button("下载压力测试报告（HTML）", data=stress_html,
                           file_name=f"stress_test_{pd.Timestamp.now():%Y%m%d}.html")


# ==================== 页面 11：一键报告 ====================
def page_report():
    st.header("一键量化分析报告")
    st.caption("汇集基金核心指标、AI 预测、持仓穿透与操作建议，支持 HTML 导出")
    code, name, fkey = cur_fund()
    format_ = st.radio("报告格式", ["HTML", "文本摘要"], horizontal=True)

    res = c_analysis(code, fkey, name, default_start(), default_end())
    if res is None:
        st.warning("无数据或样本不足，无法生成报告。"); return
    prices = res["prices"]
    metrics = res["metrics"].copy()
    _vp = metrics["估值分位"]
    metrics["估值状态"] = ind.valuation_status(_vp / 100) if _vp is not None else "数据不足"
    pred = res["pred"]
    score = scoring.composite_score(metrics, st.session_state.style)
    stress_res = stress.stress_test_fund(prices)
    advice = report.generate_operation_advice(metrics, pred, st.session_state.style,
                                              stress=stress_res, score=score)
    mc, trend = c_forecast(code, fkey, name, default_start(), default_end())
    holdings = c_holdings(code)

    if format_ == "HTML":
        html = report.build_fund_report_html(name, code, metrics, score, advice,
                                             forecasts=mc, holdings=holdings,
                                             stress=stress_res)
        st.download_button("下载 HTML 报告", data=html, file_name=f"{code}_report.html",
                           mime="text/html")
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(f"### {name}（{code}）量化分析摘要")
        st.markdown(f"**综合评分 {score['total']}（{score['rating']}，风格 {score['style']}）**")
        st.markdown("#### 操作建议")
        for k, v in advice.items():
            st.markdown(f"**{k}**：{v}")
        st.markdown("#### AI 多周期预测")
        if mc:
            fdf = pd.DataFrame([
                {"周期": f"{h}日", "上涨概率": f"{f['prob_up']*100:.0f}%",
                 "置信度": f"{f['confidence']*100:.0f}%",
                 "预期收益": f"{f.get('expected_return', 0):.2f}%"}
                for h, f in sorted(mc.items())
            ])
            st.dataframe(fdf, width="stretch", hide_index=True)
        st.markdown("#### 核心指标")
        full_table(pd.DataFrame([{"指标": k, "数值": v, "白话释义": scoring.GLOSSARY.get(k, "")}
                                 for k, v in metrics.items()]))


# ==================== 主入口 ====================
PAGES = {
    "数据驾驶舱": page_dashboard,
    "基金画像": page_profile,
    "基金筛选": page_screener,
    "定投测算": page_dca,
    "策略回测对比": page_backtest,
    "压力测试（单基金）": lambda: page_stress("单基金"),
    "一键量化报告": page_report,
    "基金对比": page_compare,
    "组合构建": page_portfolio,
    "大类资产配置": page_allocation,
    "压力测试（组合）": lambda: page_stress("组合"),
}


def main():
    if "style" not in st.session_state:
        st.session_state.style = "稳健"
    if "horizon" not in st.session_state:
        st.session_state.horizon = "中期（1-3年）"
    if "fund_class" not in st.session_state:
        st.session_state.fund_class = "场内ETF"
    page = sidebar()  # 此处会写入 st.session_state.dark_mode
    inject_css()      # 基于最新 dark_mode 注入样式，避免主题滞后一次交互
    import fundquant.charts as _charts
    _charts.set_palette(st.session_state.dark_mode)
    try:
        PAGES[page]()
    except Exception as e:
        st.error(f"页面运行时出错：{e}")
        st.exception(e)


if __name__ == "__main__":
    main()