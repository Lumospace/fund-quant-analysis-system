# -*- coding: utf-8 -*-
"""数据采集与清洗层（底层基座）。

全程基于 AKShare 真实公开数据，磁盘缓存 + 自动重试 + 异常降级。
所有接口返回标准化 DataFrame（英文列名，日期已解析，数值已清洗）。
"""
import os
import time
import json
import socket
import hashlib
import warnings
import pandas as pd
import numpy as np

import config

warnings.filterwarnings("ignore")

# 底层 HTTP 网关统一超时：AKShare 多数接口内部不带超时参数，
# 设全局 socket 默认超时可在接口无响应时快速失败，避免线程被永久阻塞。
socket.setdefaulttimeout(config.HTTP_TIMEOUT)


def _cache_path(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(config.CACHE_DIR, h + ".pkl")


def _load_cache(key: str, ttl: int):
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    try:
        if time.time() - os.path.getmtime(p) > ttl:
            return None
        return pd.read_pickle(p)
    except Exception:
        return None


def _save_cache(key: str, df):
    try:
        df.to_pickle(_cache_path(key))
    except Exception:
        pass


def cache_mtime(key: str):
    """返回磁盘缓存文件最后写入时间（本地时区字符串），无缓存返回 None。

    用于向用户如实展示"数据源时间"，避免把当前时间误标为行情时间。
    """
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
    except Exception:
        return None


def _call_ak(fn, key, ttl=config.CACHE_TTL_SECONDS, retries=3, **kwargs):
    """统一调用封装：先读缓存 -> 调用 -> 失败重试 -> 缓存写入 -> 失败降级返回空。"""
    cached = _load_cache(key, ttl)
    if cached is not None:
        return cached

    import akshare as ak
    last = None
    for i in range(retries):
        try:
            df = fn(**kwargs)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _save_cache(key, df)
                return df
            last = df
        except Exception as e:  # 接口失败，退避重试
            last = e
            time.sleep(0.6 * (i + 1))
    # 降级：尝试读取哪怕过期的缓存
    stale = _load_cache(key, ttl=10 ** 9)
    if stale is not None:
        return stale
    return pd.DataFrame()


# ==================== ETF 数据 ====================

def _etf_spot_sina():
    """新浪 ETF 分类列表：东财 push2 接口被反爬断链时的降级数据源。

    字段口径与东财一致（最新价/涨跌幅/成交量/成交额），代码去掉 sz/sh/bj 前缀。
    """
    import akshare as ak
    df = ak.fund_etf_category_sina(symbol="ETF基金")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "代码" in df.columns:
        df["代码"] = df["代码"].astype(str).str.replace(r"^(sz|sh|bj)", "", regex=True)
    return df


def get_etf_spot(force: bool = False):
    """ETF 实时行情列表。东财优先，被反爬断链时自动降级新浪源。支持强制刷新。"""
    import akshare as ak
    if force:
        clear_cache("etf_spot")
    df = _call_ak(ak.fund_etf_spot_em, "etf_spot", ttl=config.SPOT_TTL, retries=1)
    if df is None or df.empty:
        df = _call_ak(_etf_spot_sina, "etf_spot", ttl=config.SPOT_TTL)
    return _clean_etf_spot(df)


def clear_cache(key: str = None):
    """清除磁盘缓存（key 为 None 清空全部）。"""
    try:
        if key:
            p = _cache_path(key)
            if os.path.exists(p):
                os.remove(p)
        else:
            for f in os.listdir(config.CACHE_DIR):
                os.remove(os.path.join(config.CACHE_DIR, f))
    except Exception:
        pass


def _clean_etf_spot(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["代码", "名称", "最新价", "涨跌幅", "成交量", "成交额"])
    df = df.copy()
    df.rename(columns={"代码": "code", "名称": "name", "最新价": "price",
                       "涨跌幅": "pct_chg", "成交量": "volume", "成交额": "amount",
                       "IOPV实时估值": "iopv", "基金折价率": "discount"}, inplace=True)
    for c in ["price", "pct_chg", "volume", "amount", "iopv", "discount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _sina_symbol(code: str) -> str:
    """ETF 代码 -> 新浪前缀符号（5/6开头为沪市，1/15开头为深市）。"""
    code = str(code)
    if code.startswith(("5", "6")):
        return "sh" + code
    return "sz" + code


def get_etf_hist(code: str, start: str = None, end: str = None,
                 adjust: str = "qfq") -> pd.DataFrame:
    """ETF 日线历史行情（前复权）。主用东财，失败自动降级新浪源。"""
    import akshare as ak
    start = start or "20150101"
    end = end or pd.Timestamp.now().strftime("%Y%m%d")
    # 主源：东方财富
    df = _call_ak(ak.fund_etf_hist_em, f"etf_hist_em_{code}_{start}_{end}_{adjust}",
                  ttl=12 * 3600, symbol=code, period="daily",
                  start_date=start, end_date=end, adjust=adjust)
    df = _clean_ohlc(df)
    # 降级源：新浪（返回全量历史，需按区间截断）
    if df.empty:
        sdf = _call_ak(ak.fund_etf_hist_sina, f"etf_hist_sina_{code}",
                       ttl=12 * 3600, symbol=_sina_symbol(code))
        df = _clean_ohlc(sdf)
    if df.empty:
        return df
    start_ts = pd.Timestamp(start[:4] + "-" + start[4:6] + "-" + start[6:8])
    end_ts = pd.Timestamp(end[:4] + "-" + end[4:6] + "-" + end[6:8])
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)
    return df


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                       "最高": "high", "最低": "low", "成交量": "volume",
                       "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
                       "涨跌额": "chg", "换手率": "turnover"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open", "close", "high", "low", "volume", "amount",
              "amplitude", "pct_chg", "chg", "turnover"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_benchmark(name: str = "沪深300") -> pd.DataFrame:
    """返回基准指数历史行情（用 ETF 代码承载）。"""
    code = config.BENCHMARKS.get(name, "510300")
    df = get_etf_hist(code)
    return df


# ==================== 指数真实估值（PE-TTM） ====================

def get_index_pe_history(index_name: str) -> pd.DataFrame:
    """返回指定宽基指数的真实 PE-TTM 历史（date, pe），来自乐咕乐股。失败返回空。

    stock_index_pe_lg 仅支持 config.INDEX_VALUATION_MAP 中的 12 个指数，
    需先用 resolve_index_valuation 解析，避免直接把不支持的名称传入导致 KeyError。
    """
    import akshare as ak
    df = _call_ak(ak.stock_index_pe_lg, f"index_pe_{index_name}",
                  ttl=12 * 3600, symbol=index_name)
    if df is None or df.empty or "滚动市盈率" not in df.columns:
        return pd.DataFrame(columns=["date", "pe"])
    out = df[["日期", "滚动市盈率"]].copy()
    out.columns = ["date", "pe"]
    out["date"] = pd.to_datetime(out["date"])
    out["pe"] = pd.to_numeric(out["pe"], errors="coerce")
    return out.dropna(subset=["pe"]).sort_values("date").reset_index(drop=True)


def resolve_index_valuation(name) -> str:
    """按基金名称关键词解析到可获取真实估值（PE-TTM）的指数名，命中不到返回 None。"""
    n = str(name or "")
    for keyword, idx in config.INDEX_VALUATION_MAP:
        if keyword in n:
            return idx
    return None


def get_fund_pe_history(code=None, name=None) -> pd.DataFrame:
    """按基金名称解析其跟踪指数，返回真实 PE-TTM 历史；无法解析则返回空。

    code 仅作为将来可扩展的兜底入口，当前解析依赖 name 关键词。
    """
    idx = resolve_index_valuation(name)
    if idx is None:
        return pd.DataFrame(columns=["date", "pe"])
    return get_index_pe_history(idx)


# ==================== 开放式基金数据 ====================

def get_open_fund_rank(fund_type: str = "股票型") -> pd.DataFrame:
    """开放式基金排行（含阶段性收益，用于筛选/优选）。"""
    import akshare as ak
    df = _call_ak(ak.fund_open_fund_rank_em, f"openfund_rank_{fund_type}",
                  ttl=12 * 3600, symbol=fund_type)
    return _clean_open_fund_rank(df)


def _clean_open_fund_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    rename = {"基金代码": "code", "基金简称": "name", "日期": "date", "单位净值": "nav",
              "累计净值": "acc_nav", "日增长率": "d1", "近1周": "w1", "近1月": "m1",
              "近3月": "m3", "近6月": "m6", "近1年": "y1", "近2年": "y2",
              "近3年": "y3", "今年来": "ytd", "成立来": "since_inc", "手续费": "fee"}
    df.rename(columns=rename, inplace=True)
    for c in ["nav", "acc_nav", "d1", "w1", "m1", "m3", "m6", "y1", "y2",
              "y3", "ytd", "since_inc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_open_fund_hist(code: str, indicator: str = "单位净值走势") -> pd.DataFrame:
    """开放式基金历史净值走势。"""
    import akshare as ak
    df = _call_ak(ak.fund_open_fund_info_em, f"openfund_hist_{code}_{indicator}",
                  ttl=12 * 3600, symbol=code, indicator=indicator)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.rename(columns={"净值日期": "date", "单位净值": "nav",
                       "日增长率": "pct_chg"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["nav", "pct_chg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fund_code_to_name(code: str) -> str:
    """代码 -> 名称 的快速映射（优先 ETF，其次开放基金）。"""
    try:
        spot = get_etf_spot()
        hit = spot[spot["code"] == code]
        if not hit.empty and "name" in hit.columns:
            return str(hit["name"].iloc[0])
    except Exception:
        pass
    return code


# ==================== 行业/板块 与 重仓股 ====================

def classify_sector(name) -> str:
    """按基金名称关键词归集行业/板块，未命中归入「其他」。"""
    n = str(name or "")
    for sector, kws in config.SECTOR_MAP:
        if any(k in n for k in kws):
            return sector
    return "其他"


def add_sector(df: pd.DataFrame) -> pd.DataFrame:
    """给基金池补「sector」列。"""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "name" in df.columns:
        df["sector"] = df["name"].map(classify_sector)
    return df


def get_fund_holdings(code, date: str = "") -> pd.DataFrame:
    """单只基金重仓股票（前十大持仓，来自天天基金投资组合）。ETF 与场外基金通用。"""
    import akshare as ak
    df = _call_ak(ak.fund_portfolio_hold_em, f"fund_hold_{code}", ttl=12 * 3600,
                  symbol=str(code), date=date)
    if df is None or df.empty:
        return pd.DataFrame(columns=["序号", "股票代码", "股票名称", "占净值比例",
                                     "持股数", "持仓市值", "季度"])
    df = df.copy()
    for c in ["占净值比例", "持股数", "持仓市值"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ==================== 个股基本面反查（申万三级成分股，绕过不可用的单个股接口） ====================

def _sw_third_industry_info() -> pd.DataFrame:
    """申万三级行业列表（行业代码/行业名称/上级行业=二级行业名），7 天长缓存。"""
    import akshare as ak
    return _call_ak(ak.sw_index_third_info, "sw_third_info", ttl=7 * 24 * 3600, retries=2)


def _sw_second_industry_info() -> pd.DataFrame:
    """申万二级行业列表（行业名称->上级行业=一级行业名），7 天长缓存。"""
    import akshare as ak
    return _call_ak(ak.sw_index_second_info, "sw_second_info", ttl=7 * 24 * 3600, retries=2)


def _sw_third_cons(symbol: str) -> pd.DataFrame:
    """申万三级行业（symbol）成分股列表，7 天长缓存、单次重试快速降级。"""
    import akshare as ak
    return _call_ak(ak.sw_index_third_cons, f"sw_third_cons_{symbol}", ttl=7 * 24 * 3600,
                    symbol=symbol, retries=1)


def get_stock_spot_index(force: bool = False) -> pd.DataFrame:
    """全 A 股行情快照（东方财富，单次请求），提取 总市值 / 市盈率(动态)。

    作规模风格与估值风格的稳健数据源：一次请求拿全市场，避免逐股接口被远端限流。
    返回 DataFrame 列：code name pe market_cap_yi（市值单位：亿元）。6 小时缓存。
    """
    import akshare as ak
    key = "stock_spot_index"
    cols = ["code", "name", "pe", "market_cap_yi"]
    df = _call_ak(ak.stock_zh_a_spot_em, key, ttl=6 * 3600, retries=2)
    if df is None or df.empty or "代码" not in df.columns:
        return pd.DataFrame(columns=cols)

    def _col(name):
        return df[name] if name in df.columns else None

    pe = pd.to_numeric(_col("市盈率-动态"), errors="coerce")
    mc = pd.to_numeric(_col("总市值"), errors="coerce")
    name = _col("名称")
    out = pd.DataFrame({
        "code": df["代码"].astype(str),
        "name": name.astype(str) if name is not None else "",
        "pe": pe if pe is not None else np.nan,
        "market_cap_yi": (mc / 1e8) if mc is not None else np.nan,
    })
    return out.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)


def get_stock_industry_index(force: bool = False) -> pd.DataFrame:
    """申万一级行业反向映射表（code -> 一级行业），低并发 gentle 构建，7 天缓存。

    逐行业拉取三级成分股反推股票所属一级行业；低并发(3)+退避避免触发远端限流，
    成功部件按 7 天单独缓存，失败下次重跑只补缺失行业即可。
    返回 DataFrame 列：code industry。
    """
    key = "stock_industry_index"
    cols = ["code", "industry"]
    if not force:
        cached = _load_cache(key, ttl=7 * 24 * 3600)
        if cached is not None and not cached.empty:
            return cached

    third = _sw_third_industry_info()
    if third is None or third.empty or "行业代码" not in third.columns:
        return pd.DataFrame(columns=cols)

    # 三级行业代码 -> 一级行业名（经二级行业名中转）
    sec2first = {}
    second = _sw_second_industry_info()
    if second is not None and not second.empty and "行业名称" in second.columns:
        sec2first = dict(zip(second["行业名称"].astype(str), second["上级行业"].astype(str)))
    lvl1 = {}
    for _, r in third.iterrows():
        sec = str(r.get("上级行业", "") or "")
        lvl1[str(r["行业代码"])] = sec2first.get(sec, sec) or "未知"

    def _one(symbol):
        cons = _sw_third_cons(symbol)
        if cons is None or cons.empty or "股票代码" not in cons.columns:
            return None
        c = cons[["股票代码"]].copy()
        c.columns = ["code"]
        c["code"] = c["code"].astype(str).str.split(".").str[0]
        c["industry"] = lvl1.get(symbol, "未知")
        return c

    symbols = third["行业代码"].astype(str).tolist()
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    parts = []
    ex = ThreadPoolExecutor(max_workers=3)
    try:
        deadline = _time.time() + config.LOOKTHROUGH_BUDGET
        futures = [ex.submit(_one, s) for s in symbols]
        for fut in as_completed(futures):
            if _time.time() > deadline:
                break  # 超时提前结束，剩余行业留给下次缓存补齐
            try:
                part = fut.result()
            except Exception:
                part = None
            if part is not None and not part.empty:
                parts.append(part)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    out = pd.DataFrame(columns=cols)
    if parts:
        out = pd.concat(parts, ignore_index=True)
        out = out.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
        _save_cache(key, out)
    return out


# ==================== 场外基金实时估值 ====================

def get_otc_estimation() -> pd.DataFrame:
    """场外基金实时估值（估算净值 + 估算增长率，官方按重仓股实时行情估算）。"""
    import akshare as ak
    df = _call_ak(ak.fund_value_estimation_em, "otc_value_est", ttl=config.SPOT_TTL, symbol="全部")
    return _clean_otc_estimation(df)


def _clean_otc_estimation(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "price", "pct_chg"])
    df = df.copy()
    df = df.rename(columns={"基金代码": "code", "基金名称": "name"})
    est_col = next((c for c in df.columns if "估算值" in c), None)
    chg_col = next((c for c in df.columns if "估算增长率" in c), None)
    keep = ["code", "name"] + [c for c in [est_col, chg_col] if c]
    df = df[keep].copy()
    if est_col:
        df = df.rename(columns={est_col: "price"})
    else:
        df["price"] = None
    if chg_col:
        df = df.rename(columns={chg_col: "pct_chg"})
    else:
        df["pct_chg"] = None
    # 从原始列名（如 2026-09-18-估算数据-估算值）提取估值日期，用于判断是否当日更新
    _m = __import__("re").search(r"(\d{4}-\d{2}-\d{2})", (est_col or chg_col or ""))
    df["est_date"] = _m.group(1) if _m else ""
    for c in ["price", "pct_chg"]:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.replace("%", "", regex=False)
                     .str.replace(",", "", regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ==================== 统一基金池 / 历史（场内&场外） ====================

def _to_ts(ymd):
    if not ymd:
        return None
    ymd = str(ymd)
    if len(ymd) != 8 or not ymd.isdigit():
        return None
    try:
        return pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
    except Exception:
        return None


def get_otc_universe() -> pd.DataFrame:
    """完整场外基金池：全量基金名录（含债券/货币/QDII等）+ 可用的实时估值。"""
    import akshare as ak
    names = _call_ak(ak.fund_name_em, "fund_name_em", ttl=24 * 3600)
    if names is None or names.empty:
        return get_otc_estimation()
    names = names.copy().rename(columns={"基金代码": "code", "基金简称": "name",
                                         "基金类型": "fund_type"})
    names = names[["code", "name", "fund_type"]].copy()
    names["code"] = names["code"].astype(str)
    est = get_otc_estimation()
    if est is not None and not est.empty:
        est = est.copy()
        est["code"] = est["code"].astype(str)
        names = names.merge(est[["code", "price", "pct_chg"]], on="code", how="left")
    else:
        names["price"] = np.nan
        names["pct_chg"] = np.nan
    names = names.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    names["sector"] = names["fund_type"].astype(str)
    return names


def get_fund_universe(ftype: str = "etf") -> pd.DataFrame:
    """统一基金池：etf 返回实时行情，otc 返回完整名录+可用估值，均含 sector。"""
    if ftype == "otc":
        return get_otc_universe()
    df = add_sector(get_etf_spot())
    return df


def get_fund_list(ftype: str = "etf") -> pd.DataFrame:
    """轻量基金列表（仅代码/名称/板块，不含实时价格），供选择器使用，长缓存。"""
    import akshare as ak
    if ftype == "otc":
        names = _call_ak(ak.fund_name_em, "fund_name_em", ttl=24 * 3600)
        if names is None or names.empty:
            return pd.DataFrame()
        names = names.rename(columns={"基金代码": "code", "基金简称": "name",
                                      "基金类型": "fund_type"})
        df = names[["code", "name", "fund_type"]].copy()
        df["code"] = df["code"].astype(str)
        df["sector"] = df["fund_type"].astype(str)
        return df.drop_duplicates(subset=["code"]).reset_index(drop=True)
    df = add_sector(get_etf_spot())
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in ["code", "name", "sector"] if c in df.columns]
    return df[cols].reset_index(drop=True)


def get_fund_hist(code, ftype: str = "etf", start: str = None, end: str = None,
                  adjust: str = "qfq") -> pd.DataFrame:
    """统一历史行情：etf 走日线（close），otc 走净值（nav）。"""
    if ftype == "otc":
        h = get_open_fund_hist(code)
        if h is None or h.empty:
            return h
        ts_s = _to_ts(start)
        ts_e = _to_ts(end)
        if ts_s is not None:
            h = h[h["date"] >= ts_s]
        if ts_e is not None:
            h = h[h["date"] <= ts_e]
        return h.reset_index(drop=True)
    return get_etf_hist(code, start, end, adjust)


if __name__ == "__main__":
    # 自检
    s = get_etf_spot()
    print("ETF spot:", s.shape, s.columns.tolist()[:8])
    h = get_etf_hist("510300", "20240101", "20240601")
    print("ETF hist:", h.shape, h.columns.tolist())
    r = get_open_fund_rank("股票型")
    print("Open fund rank:", r.shape, r.columns.tolist()[:10])