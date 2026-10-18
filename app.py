import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, mannwhitneyu, norm as _norm

st.set_page_config(page_title="ZaloPay Test", layout="centered", initial_sidebar_state="expanded")

# ── Visitor logging ────────────────────────────────────────────────────────────
@st.cache_resource
def _get_log_sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(st.secrets["sheet"]["id"]).sheet1

def _log_visit():
    if st.session_state.get("_visit_logged"):
        return
    try:
        sheet = _get_log_sheet()
        h   = st.context.headers
        ip  = h.get("X-Forwarded-For", h.get("X-Real-Ip", "unknown")).split(",")[0].strip()
        ua  = h.get("User-Agent", "unknown")
        now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, ip, ua], value_input_option="RAW")
        st.session_state["_visit_logged"] = True
    except Exception:
        pass

_log_visit()

st.markdown("""
<style>
    .block-container { max-width: 860px; padding-top: 2rem; }
    .question-label { font-size: 0.78rem; font-weight: 600; color: #888;
                      text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {visibility: visible !important; display: flex !important;}
    [data-testid="stSidebar"] {visibility: visible !important;}
    [data-testid="stMetric"] { text-align: center !important; }
    [data-testid="stMetricLabel"] { justify-content: center !important; display: flex !important; }
    [data-testid="stMetricValue"] { justify-content: center !important; display: flex !important; }
    [data-testid="stMetricDelta"] { justify-content: center !important; display: flex !important; }
    [data-testid="stMetricDelta"] > div { margin: 0 auto !important; }
    div[data-testid="stMetric"] > label { text-align: center !important; width: 100%; }
    div[data-testid="stMetric"] > div { justify-content: center !important; }
    [role="tablist"] {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: white;
        padding-top: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("data.csv")
    df["trxn_date"]  = pd.to_datetime(df["trxn_date"])
    df["trxn_month"] = df["trxn_date"].dt.to_period("M")
    df["cohort"]     = df.groupby("user_id")["trxn_date"].transform("min").dt.to_period("M")
    df["carrier"]    = df["carrier"].replace("Other services", "Other Services")

    active = df.groupby(["user_id","trxn_month"]).size().reset_index()[["user_id","trxn_month"]]
    active = active.merge(df[["user_id","cohort"]].drop_duplicates(), on="user_id")
    active["month_number"] = (active["trxn_month"] - active["cohort"]).apply(lambda x: x.n)
    cohort_size = active[active["month_number"]==0].groupby("cohort")["user_id"].nunique()
    pivot = active.groupby(["cohort","month_number"])["user_id"].nunique().unstack()
    pivot = pivot.div(cohort_size, axis=0)

    m0    = df[df["trxn_month"] == df["cohort"]]
    m1uid = set(df[df["trxn_month"] == df["cohort"].map(lambda c: c+1)]["user_id"])
    m0s   = m0.groupby("user_id").agg(
        n_txn      =("transaction_id","count"),
        gmv        =("trxn_amount","sum"),
        med_aov    =("trxn_amount","median"),
        total_limit=("total_limit","first"),
        n_carriers =("carrier","nunique"),
    )
    m0s["util"]     = (m0s["gmv"] / m0s["total_limit"]).clip(0,1)
    m0s["fee_pct"]  = 20_000 / m0s["gmv"]
    m0s["retained"] = m0s.index.isin(m1uid)

    # User-level M0 labels for filtering
    m0_carrier = m0.groupby("user_id")["carrier"].agg(lambda x: x.mode()[0]).rename("primary_carrier")
    m0_channel  = m0.groupby("user_id")["package_code"].agg(lambda x: x.mode()[0]).rename("primary_channel")
    user_labels = m0_carrier.to_frame().join(m0_channel).reset_index()  # user_id as column

    # Extended active with GMV per user-month
    gmv_um = df.groupby(["user_id","trxn_month"])["trxn_amount"].sum().rename("gmv_month").reset_index()
    active_ext = active.merge(user_labels, on="user_id", how="left").merge(gmv_um, on=["user_id","trxn_month"], how="left")

    return df, active, active_ext, pivot, cohort_size, m0s, m1uid, user_labels

df, active, active_ext, pivot, cohort_size, m0s, m1uid, user_labels = load()

total   = df["user_id"].nunique()
n_churn = total - len(m1uid)
n_ret   = len(m1uid)
pct_c   = n_churn / total

ALL_CARRIERS = sorted(user_labels["primary_carrier"].dropna().unique())
ALL_CHANNELS = sorted(user_labels["primary_channel"].dropna().unique())

_all_cohorts  = sorted(active_ext["cohort"].unique())
ALL_YEARS     = sorted({int(str(p)[:4]) for p in _all_cohorts})
ALL_MONTHS    = [pd.Period(p, freq="M").strftime("%m/%Y") for p in _all_cohorts]

def clean_fig(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="sans-serif", size=12, color="#333"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("ZaloPay Test")
st.caption("BI Analyst Take-home · BNPL Retention & Churn Analysis")
st.divider()

# ── Sidebar filters - apply toàn bộ chart ─────────────────────────────────────
with st.sidebar:
    st.markdown("**Bộ lọc**")
    sel_year    = st.multiselect("Năm", ALL_YEARS, placeholder="Tất cả")
    sel_month   = st.multiselect("Tháng/Năm", ALL_MONTHS, placeholder="Tất cả")
    sel_carrier = st.multiselect("Carrier", ALL_CARRIERS, placeholder="Tất cả")
    sel_channel = st.multiselect("Channel", ALL_CHANNELS, placeholder="Tất cả")

# Global filtered dataset (used across all tabs)
filt = active_ext.copy()
if sel_year:
    filt = filt[filt["cohort"].astype(str).str[:4].astype(int).isin(sel_year)]
if sel_month:
    _sel_periods = {pd.Period(m, freq="M") for m in sel_month}
    filt = filt[filt["cohort"].isin(_sel_periods)]
if sel_carrier:
    filt = filt[filt["primary_carrier"].isin(sel_carrier)]
if sel_channel:
    filt = filt[filt["primary_channel"].isin(sel_channel)]
cs_filt = filt[filt["month_number"] == 0].groupby("cohort")["user_id"].nunique()

# Filtered m0s for Part 2/3
filt_users = set(filt["user_id"].unique())
m0s_filt = m0s[m0s.index.isin(filt_users)]

tabs = st.tabs(["Q1a", "Q1b", "Q1c", "Part 2 - Churn", "Part 3 - So What?"])

# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Define 'retention' in a way that makes sense for this product and dataset.")
    st.markdown("""
Sản phẩm **Buy Now Pay Later** của **ZaloPay** (Tài khoản Trả Sau) là sản phẩm liên kết với CIMB và LOTTE Finance là lender, ZaloPay đóng vai trò là payment intermediary và distribution layer.

Do vậy để xác định metric thật sự có ý nghĩa đối với Zalo Pay, cần nhìn nhận nguồn thu phân theo vai trò:

""")
    st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:14px">
<thead><tr style="background:#f3f4f6">
  <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb">Nguồn thu</th>
  <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb">Bên nhận</th>
</tr></thead>
<tbody>
<tr style="background:#f0fdf4">
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">Merchant fee (mỗi giao dịch)</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#15803d;font-weight:600">ZaloPay</td>
</tr>
<tr style="background:#fefce8">
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">Phí dịch vụ 20.000đ/kỳ</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#92400e">Đối tác / chia tỉ lệ (Chưa có thông tin xác minh)</td>
</tr>
<tr style="background:#f9fafb">
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">Lãi suất quá hạn</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#6b7280">CIMB / LOTTE Finance</td>
</tr>
<tr style="background:#f9fafb">
  <td style="padding:10px 14px">Phí trễ hạn</td>
  <td style="padding:10px 14px;color:#6b7280">CIMB / LOTTE Finance</td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)
    st.caption("Nguồn: ZaloPay - Tài khoản Trả Sau thanh toán được những gì?")
    st.markdown("""

Như vậy, Nguồn doanh thu thu trực tiếp của ZaloPay gắn với giao dịch mua. Từ đó, để theo dõi và tối ưu retention rate cho sản phẩm này, cần xác định 2 thành phần:

- **Active user definition**: tiêu chí xác định user đang sử dụng sản phẩm
- **Measurement period**: kỳ hạn đo lường phù hợp với chu kỳ sản phẩm

#### 1. Active user


""")
    st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:14px">
<thead><tr style="background:#f3f4f6">
  <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb">Định nghĩa</th>
  <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb">Trạng thái user</th>
  <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb">Ý nghĩa với BNPL</th>
</tr></thead>
<tbody>
<tr style="background:#f0fdf4">
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">User có ít nhất 1 giao dịch mua trong tháng</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb"><strong>User tạo doanh thu trực tiếp cho ZaloPay</strong></td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#15803d;font-weight:600">Cao</td>
</tr>
<tr style="background:#fefce8">
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">User có Outstanding > 0</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb">User còn dư nợ chưa trả</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#92400e">Thấp - revenue thuộc lender</td>
</tr>
<tr style="background:#fef2f2">
  <td style="padding:10px 14px">User mở mới hồ sơ BNPL</td>
  <td style="padding:10px 14px">User thực hiện đăng ký, chưa sử dụng</td>
  <td style="padding:10px 14px;color:#991b1b">Chưa tạo doanh thu trực tiếp</td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)
    st.info("Active User = có ít nhất ≥1 giao dịch mua trong kỳ.")
    st.markdown("""
#### 2. Measurement period

- Billing cycle của sản phẩm BNPL theo tháng - chu kỳ sao kê, chốt nợ và thu phí dịch vụ 20K đều lặp lại hàng tháng. ([Nguồn: ZaloPay - Huỷ tài khoản Trả Sau](https://zalopay.vn/huy-tai-khoan-tra-sau-7073))
- Tháng cũng trùng với chu kỳ tài chính của user - nhận lương, trả hóa đơn định kỳ.

""")
    st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:14px">
<thead><tr style="background:#f3f4f6">
  <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb;width:80px">Kỳ hạn</th>
  <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb">Đánh giá</th>
</tr></thead>
<tbody>
<tr style="background:#f9fafb">
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#6b7280;font-weight:600">Tuần</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#6b7280">Khá ngắn - đối với chi trả hoá đơn, chi tiêu có thể user chỉ sử dụng 1 tháng/lần</td>
</tr>
<tr style="background:#f0fdf4">
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#15803d;font-weight:600">Tháng</td>
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#15803d">Tương đương một chu kỳ thanh toán, đủ để nhanh chóng phát hiện user rời bỏ sản phẩm</td>
</tr>
<tr style="background:#f9fafb">
  <td style="padding:10px 14px;color:#6b7280;font-weight:600">Quý</td>
  <td style="padding:10px 14px;color:#6b7280">Khá trễ - có thể user đã rời đi 1-2 tháng trước đó</td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)

    st.info("Measurement period = tháng.")

    st.markdown("#### Kết luận - Định nghĩa Retention")
    st.info(r"""$$Retention\ M(k)=\frac{\text{Số user trong cohort có} \ge 1\ \text{giao dịch mua trong tháng thứ } k}{\text{Tổng số user thuộc cohort đó}}\times 100\%$$""")
    st.caption("Trong đó: cohort = nhóm user theo tháng phát sinh giao dịch mua đầu tiên. Định nghĩa Churn threshold xem chi tiết tại tab **Churn Analysis**.")

# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Build a cohort retention analysis. Patterns & anomalies?")

    # Dynamic M1 stats
    _piv_ret = filt.groupby(["cohort","month_number"])["user_id"].nunique().unstack().div(cs_filt, axis=0)
    _m1_col  = _piv_ret[1] if 1 in _piv_ret.columns else pd.Series(dtype=float)
    _m1_avg  = _m1_col.mean()  if len(_m1_col) else float("nan")
    _m1_min  = _m1_col.min()   if len(_m1_col) else float("nan")
    _m1_max  = _m1_col.max()   if len(_m1_col) else float("nan")
    _n_coh   = len(_piv_ret)
    _m1_pp   = (_m1_max - _m1_min) * 100 if len(_m1_col) else float("nan")
    # New user trend
    _new_mon_all = filt[filt["month_number"]==0].groupby("trxn_month")["user_id"].nunique()
    _new_peak_val = int(_new_mon_all.max()) if len(_new_mon_all) else 0
    _new_peak_mon = _new_mon_all.idxmax() if len(_new_mon_all) else None
    _new_last_val = int(_new_mon_all.iloc[-1]) if len(_new_mon_all) else 0
    _new_drop_pct = (_new_last_val - _new_peak_val) / _new_peak_val if _new_peak_val else float("nan")
    _new_peak_lbl = pd.Period(_new_peak_mon, freq="M").strftime("%m/%Y") if _new_peak_mon else "–"

    _m1_avg_fmt = f"{_m1_avg:.1%}" if not np.isnan(_m1_avg) else "–"
    _m1_rng_fmt = f"{_m1_min:.1%}–{_m1_max:.1%}" if not np.isnan(_m1_min) else "–"
    _m1_pp_fmt  = f"{_m1_pp:.1f}%" if not np.isnan(_m1_pp) else "–"

    st.write(f"Về tổng quan, M1 retention rate trung bình đạt **{_m1_avg_fmt}**, ổn định qua {_n_coh} cohorts ({_m1_rng_fmt}) với dải dao động chỉ ~{_m1_pp_fmt}.")

    # Pre-compute YoY cards
    _mau_mon  = filt.groupby("trxn_month")["user_id"].nunique()
    _gmv_mon  = filt.groupby("trxn_month")["gmv_month"].sum()
    _new_mon  = filt[filt["month_number"] == 0].groupby("trxn_month")["user_id"].nunique()
    _h24 = [p for p in _mau_mon.index if str(p)[:4] == "2024" and 1 <= int(str(p)[5:7]) <= 6]
    _h25 = [p for p in _mau_mon.index if str(p)[:4] == "2025" and 1 <= int(str(p)[5:7]) <= 6]
    _new_24 = float(_new_mon[[p for p in _h24 if p in _new_mon.index]].mean()) if _h24 else float("nan")
    _new_25 = float(_new_mon[[p for p in _h25 if p in _new_mon.index]].mean()) if _h25 else float("nan")
    _new_yoy = (_new_25 - _new_24) / _new_24 if _new_24 else float("nan")
    _gmv_24 = float(_gmv_mon[_h24].sum()) if _h24 else float("nan")
    _gmv_25 = float(_gmv_mon[_h25].sum()) if _h25 else float("nan")
    _gmv_yoy = (_gmv_25 - _gmv_24) / _gmv_24 if _gmv_24 else float("nan")
    _mau_24 = float(_mau_mon[_h24].mean()) if _h24 else float("nan")
    _mau_25 = float(_mau_mon[_h25].mean()) if _h25 else float("nan")
    _mau_yoy = (_mau_25 - _mau_24) / _mau_24 if _mau_24 else float("nan")
    _gmv_per_user_24 = _gmv_24 / (_mau_24 * 6) if _mau_24 else float("nan")
    _gmv_per_user_25 = _gmv_25 / (_mau_25 * 6) if _mau_25 else float("nan")
    _gpu_yoy = (_gmv_per_user_25 - _gmv_per_user_24) / _gmv_per_user_24 if _gmv_per_user_24 else float("nan")
    _latest_new = int(_new_mon.iloc[-1]) if len(_new_mon) else 0

    from plotly.subplots import make_subplots

    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _mc1.metric("New users (avg H1)", f"{_new_25:,.0f}" if not np.isnan(_new_25) else "–",
                f"{_new_yoy:+.0%} YoY" if not np.isnan(_new_yoy) else "–")
    _mc2.metric("MAU (avg H1)", f"{_mau_25:,.0f}" if not np.isnan(_mau_25) else "–",
                f"{_mau_yoy:+.0%} YoY" if not np.isnan(_mau_yoy) else "–")
    _mc3.metric("GMV (H1 2025)", f"{_gmv_25/1e9:.1f}B" if not np.isnan(_gmv_25) else "–",
                f"{_gmv_yoy:+.0%} YoY" if not np.isnan(_gmv_yoy) else "–")
    _mc4.metric("GMV / user (avg H1)", f"{_gmv_per_user_25/1000:.0f}K" if not np.isnan(_gmv_per_user_25) else "–",
                f"{_gpu_yoy:+.0%} YoY" if not np.isnan(_gpu_yoy) else "–")

    st.markdown("<div style='text-align:center;font-size:14px;margin-bottom:4px'>Chọn loại Metric cho Cohort Matrix</div>", unsafe_allow_html=True)
    _ml, _mc, _mr = st.columns([1, 2, 1])
    with _mc:
        metric = st.selectbox("Chọn loại Metric cho Cohort Matrix",
                              ["Retention Rate (%)", "Số user", "Chi tiêu/user (mean)", "Chi tiêu/user (median)", "GMV (Gross Merchandise Volume)"],
                              index=0, key="q1b_metric", label_visibility="collapsed")

    # ── Compute metric pivot ──────────────────────────────────────────────────
    if metric == "Retention Rate (%)":
        piv = filt.groupby(["cohort","month_number"])["user_id"].nunique().unstack().div(cs_filt, axis=0)
        fmt_fn = lambda v: f"{v:.0%}" if not np.isnan(v) else ""
        cscale = [[0,"#f9f9f9"],[0.5,"#bfdbfe"],[1,"#1d4ed8"]]
        zmin, zmax = 0, 1
    elif metric == "Số user":
        piv = filt.groupby(["cohort","month_number"])["user_id"].nunique().unstack()
        fmt_fn = lambda v: f"{int(v):,}" if not np.isnan(v) else ""
        cscale = [[0,"#f9f9f9"],[0.5,"#bfdbfe"],[1,"#1d4ed8"]]
        zmin, zmax = None, None
    elif metric == "Chi tiêu/user (mean)":
        gmv_piv = filt.groupby(["cohort","month_number"])["gmv_month"].sum().unstack()
        usr_piv = filt.groupby(["cohort","month_number"])["user_id"].nunique().unstack()
        piv = gmv_piv / usr_piv / 1000
        fmt_fn = lambda v: f"{v:.0f}K" if not np.isnan(v) else ""
        cscale = [[0,"#f9f9f9"],[0.5,"#bfdbfe"],[1,"#1d4ed8"]]
        zmin, zmax = None, None
    elif metric == "Chi tiêu/user (median)":
        piv = filt.groupby(["cohort","month_number"])["gmv_month"].median().unstack() / 1000
        fmt_fn = lambda v: f"{v:.0f}K" if not np.isnan(v) else ""
        cscale = [[0,"#f9f9f9"],[0.5,"#bfdbfe"],[1,"#1d4ed8"]]
        zmin, zmax = None, None
    else:  # GMV
        piv = filt.groupby(["cohort","month_number"])["gmv_month"].sum().unstack() / 1e6
        fmt_fn = lambda v: f"{v:.1f}M" if not np.isnan(v) else ""
        cscale = [[0,"#f9f9f9"],[0.5,"#bfdbfe"],[1,"#1d4ed8"]]
        zmin, zmax = None, None

    cols = [c for c in piv.columns if c <= 12]
    pd_data = piv[cols]
    z_ret = pd_data.values
    text_ret = [[fmt_fn(v) for v in row] for row in z_ret]
    ylabels  = [c.to_timestamp().strftime("%b %Y") for c in pd_data.index]
    sizes    = [cs_filt.get(c, 0) for c in pd_data.index]

    s_arr = np.array(sizes, dtype=float)
    z_size = (s_arr / max(s_arr.max(), 1)).reshape(-1, 1)
    text_size = [[f"{s:,}"] for s in sizes]

    # Average row
    pct_piv_raw = filt.groupby(["cohort","month_number"])["user_id"].nunique().unstack().div(cs_filt, axis=0)
    avg_row_vals = pct_piv_raw[cols].mean(skipna=True)
    avg_size_val = np.array([s_arr.mean()], dtype=float) / max(s_arr.max(), 1)
    avg_ylabels = ylabels + ["Avg"]

    # Append avg row to z_ret, text_ret, z_size, text_size
    avg_z_ret = list(z_ret) + [[avg_row_vals.get(c, float("nan")) for c in cols]]
    avg_text_ret = list(text_ret) + [[fmt_fn(avg_row_vals.get(c, float("nan"))) for c in cols]]
    avg_z_size = list(z_size) + [[float(avg_size_val[0])]]
    avg_text_size = list(text_size) + [[f"{int(s_arr.mean()):,}"]]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        column_widths=[0.08, 0.92], horizontal_spacing=0.01)
    fig.add_trace(go.Heatmap(
        z=avg_z_size, x=["New users"], y=avg_ylabels,
        text=avg_text_size, texttemplate="%{text}",
        colorscale=[[0,"#f0fdf4"],[1,"#16a34a"]],
        zmin=0, zmax=1, showscale=False,
        hovertemplate="%{y}: %{text} users<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=avg_z_ret, x=[f"M{i}" for i in cols], y=avg_ylabels,
        text=avg_text_ret, texttemplate="%{text}",
        colorscale=cscale, zmin=zmin, zmax=zmax, showscale=False,
        hovertemplate="Cohort: %{y}<br>%{x}: %{text}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(height=560, margin=dict(l=10,r=20,t=30,b=30),
                      paper_bgcolor="white", font=dict(size=10))
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    st.markdown("<div style='text-align:center;font-size:18px;font-weight:700;margin-bottom:4px'>Cohort Retention Matrix</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


    pct_piv = filt.groupby(["cohort","month_number"])["user_id"].nunique().unstack().div(cs_filt, axis=0)

    st.markdown("**Average retention curve - drop tại từng tháng**")
    avg_all = pct_piv[[c for c in pct_piv.columns if 0 <= c <= 12]].mean(skipna=True)
    x_wf, y_wf, measure, text_wf, hover_wf = [], [], [], [], []

    x_wf.append("M0"); y_wf.append(avg_all.get(0, 1.0))
    measure.append("absolute"); text_wf.append("100%"); hover_wf.append("M0: avg 100%")

    prev = avg_all.get(0, 1.0)
    for m in [c for c in avg_all.index if c >= 1]:
        cur  = avg_all[m]
        drop = cur - prev
        x_wf.append(f"M{m}"); y_wf.append(drop)
        measure.append("relative")
        text_wf.append(f"{drop:+.0%}")
        hover_wf.append(f"M{m}: avg {cur:.1%} | drop {drop:+.1%}")
        prev = cur

    fig_wf = go.Figure(go.Waterfall(
        x=x_wf, y=y_wf, measure=measure,
        text=text_wf, textposition="outside",
        textfont=dict(size=10),
        decreasing=dict(marker_color="#ef4444"),
        increasing=dict(marker_color="#22c55e"),
        totals=dict(marker_color="#374151"),
        connector=dict(line=dict(color="#e5e7eb", width=1)),
        customdata=hover_wf,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig_wf.update_layout(
        height=300, margin=dict(l=20,r=20,t=10,b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        xaxis=dict(showgrid=False),
        showlegend=False, font=dict(size=11),
    )
    st.plotly_chart(fig_wf, use_container_width=True, config={"displayModeBar": False})

    # ── Compute supporting stats ──────────────────────────────────────────────
    _m1_uid = set(filt[filt["month_number"] == 1]["user_id"]) & filt_users
    _m2_uid = set(filt[filt["month_number"] == 2]["user_id"]) & filt_users
    _churned_m1 = filt_users - _m1_uid
    _returning_m2 = _churned_m1 & _m2_uid
    _return_rate = len(_returning_m2) / len(_churned_m1) if _churned_m1 else 0
    _m2_ret = float(avg_all.get(2, float("nan")))
    _m3_ret = float(avg_all.get(3, float("nan")))

    st.caption(f"Drop lớn nhất ở M1 (−26%) - đây là điểm can thiệp quan trọng nhất. M2 giảm thêm {_m2_ret - float(avg_all.get(1,0)):.0%}, sau đó chậm dần nhưng tiếp tục giảm đến M12 (~5%) - không hình thành nhóm user gắn bó ổn định.")

    st.markdown("**Patterns & Anomalies**")
    st.markdown(f"""
**Retention % ổn định nhưng số lượng tuyệt đối đang giảm.**
M1 retention dao động {_m1_rng_fmt} qua {_n_coh} cohorts - product không xấu đi. Tuy nhiên, cùng lúc đó:
- Số user mới giảm từ ~{_new_peak_val:,} xuống ~{_new_last_val:,}/tháng ({_new_drop_pct:.0%} từ {_new_peak_lbl}) - số user retained tuyệt đối giảm tương ứng
- Tổng GMV co lại theo base user, dù chi tiêu/user tháng đầu ổn định
- Quick Ratio < 1 kể từ khi acquisition đạt đỉnh - base user đang giảm mỗi tháng

Chỉ nhìn vào % retention sẽ bỏ sót bức tranh thực: số user và GMV đang co lại từng tháng.

**Churn tập trung ở M1** (−26%). Từ M2 trở đi tốc độ giảm chậm dần nhưng không ổn định - không hình thành nhóm user gắn bó lâu dài. Ai rời thì rời hẳn: tỷ lệ quay lại chỉ {_return_rate:.1%} ({len(_returning_m2):,}/{len(_churned_m1):,} user).

**Không có cohort nào nổi bật.** Dải {_m1_pp_fmt} qua {_n_coh} cohorts - churn không gắn với đợt acquisition hay sự kiện cụ thể nào, mà nhất quán trên toàn bộ dữ liệu.
""")

    st.markdown("**Giả thuyết nguyên nhân:**")
    st.write("Phí 20K/tháng cố định tạo gánh nặng không đồng đều - user giao dịch ít chịu tỷ lệ phí/GMV cao, khiến sản phẩm không còn đáng dùng. Kiểm định định lượng ở tab **Churn Analysis**.")

    st.info(f"Retention % ổn định ({_m1_avg_fmt}) nhưng tổng user và GMV vẫn giảm vì lượng user mới giảm {abs(_new_drop_pct):.0%} so với đỉnh. User đã rời hầu như không quay lại ({_return_rate:.1%}), nên điểm can thiệp thực tế duy nhất là cải thiện activation ngay trong tháng đầu (M0).")

# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader('"Users who complete at least N transactions in M0 retain at a higher rate in M1." Validate or disprove. What is N?')

    m0_txn = df[df["trxn_month"]==df["cohort"]].groupby("user_id").size().rename("m0_txn")
    m0_all = m0_txn.reset_index()
    m0_all = m0_all[m0_all["user_id"].isin(filt_users)]
    m1uid_filt = m1uid & filt_users
    m0_all["retained"] = m0_all["user_id"].isin(m1uid_filt).astype(int)
    freq_raw = m0_all.groupby("m0_txn").agg(n=("user_id","count"), ret=("retained","mean")).query("n>=20")
    # Gộp 8+ thành 1 bucket
    _above8 = m0_all[m0_all["m0_txn"] > 8]
    _row8p = pd.DataFrame({"n": [len(_above8)], "ret": [_above8["retained"].mean()]}, index=[9])
    freq = pd.concat([freq_raw[freq_raw.index <= 8], _row8p])
    # Display labels: numeric 9 → "8+" on tick axis
    freq_xlabels = ["8+" if i == 9 else str(i) for i in freq.index]

    THRESHOLD = 4
    p_high   = m0_all[m0_all["m0_txn"]>=THRESHOLD]["retained"].mean()
    p_low    = m0_all[m0_all["m0_txn"]<THRESHOLD]["retained"].mean()
    n_low    = int((m0_all["m0_txn"]<THRESHOLD).sum())
    n_high   = int((m0_all["m0_txn"]>=THRESHOLD).sum())
    total_u  = len(m0_all)
    pct_below = n_low / total_u
    avg_ret  = m0_all["retained"].mean()

    # GMV / AOV per txn bucket
    m0_gmv = df[df["trxn_month"]==df["cohort"]].groupby("user_id").agg(
        gmv=("trxn_amount","sum"), med_aov=("trxn_amount","median")
    )
    m0_all2 = m0_all.merge(m0_gmv, on="user_id", how="left")
    m0_all2["fee_pct"] = 20_000 / m0_all2["gmv"]
    drv = m0_all2.groupby("m0_txn").agg(
        med_gmv=("gmv","median"), med_aov=("med_aov","median"),
        med_fee_pct=("fee_pct","median")
    )

    med_txn  = m0_all["m0_txn"].median()
    mean_txn = m0_all["m0_txn"].mean()

    # Combined chart: user distribution (bars) + retention line (dual axis)
    fig_combo = go.Figure()
    fig_combo.add_trace(go.Bar(
        x=list(freq.index), y=freq["n"],
        name="Số user",
        marker_color=["#fca5a5" if i < THRESHOLD else "#a3c4f3" for i in freq.index],
        yaxis="y",
        customdata=freq_xlabels,
        hovertemplate="M0 txn=%{customdata}<br>Số user=%{y:,}<extra></extra>",
    ))
    # Chỉ label ở điểm quan trọng: x=1 (thấp nhất) và x=THRESHOLD (nhảy vọt)
    _label_text = [f"{v:.1%}" if i in [1, THRESHOLD] else "" for i, v in zip(freq.index, freq["ret"])]
    fig_combo.add_trace(go.Scatter(
        x=list(freq.index), y=freq["ret"],
        name="M1 Retention",
        mode="lines+markers+text",
        line=dict(color="#1d4ed8", width=2.5),
        marker=dict(size=7, color="#1d4ed8"),
        text=_label_text,
        textposition="top center",
        textfont=dict(size=10, color="#1d4ed8"),
        yaxis="y2",
        customdata=freq_xlabels,
        hovertemplate="M0 txn=%{customdata}<br>M1 Retention=%{y:.1%}<extra></extra>",
    ))
    # Benchmark line: avg retention (trục y2)
    fig_combo.add_hline(y=avg_ret, line_dash="dot", line_color="#f97316", line_width=1.5,
                        annotation_text=f"Avg Retention {avg_ret:.1%}",
                        annotation_position="right",
                        annotation_font_color="#f97316", yref="y2")
    # N=4 vline - reference chính
    fig_combo.add_vline(x=THRESHOLD - 0.5, line_dash="dash", line_color="#374151", line_width=1.5,
                        annotation_text=f"N = {THRESHOLD}",
                        annotation_position="top", annotation_font_color="#374151")
    fig_combo.update_layout(
        height=360, margin=dict(l=40, r=70, t=60, b=50),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(size=11),
        yaxis=dict(title="Số user", showgrid=True, gridcolor="#f0f0f0",
                   zeroline=False, tickformat=","),
        yaxis2=dict(title="M1 Retention", overlaying="y", side="right",
                    tickformat=".0%", showgrid=False, zeroline=False,
                    range=[max(0, freq["ret"].min() - 0.08), min(1.0, freq["ret"].max() + 0.06)]),
        xaxis=dict(title="Số giao dịch M0", showgrid=False,
                   tickvals=list(freq.index), ticktext=freq_xlabels),
        barmode="relative",
    )
    # Plain version (no color highlight) for Section 1
    fig_combo_plain = go.Figure()
    fig_combo_plain.add_trace(go.Bar(
        x=list(freq.index), y=freq["n"],
        name="Số user",
        marker_color="#a3c4f3",
        yaxis="y",
        customdata=freq_xlabels,
        hovertemplate="M0 txn=%{customdata}<br>Số user=%{y:,}<extra></extra>",
    ))
    fig_combo_plain.add_trace(go.Scatter(
        x=list(freq.index), y=freq["ret"],
        name="M1 Retention",
        mode="lines+markers+text",
        line=dict(color="#1d4ed8", width=2.5),
        marker=dict(size=7, color="#1d4ed8"),
        text=[f"{v:.1%}" if i in [1, THRESHOLD] else "" for i, v in zip(freq.index, freq["ret"])],
        textposition="top center",
        textfont=dict(size=10, color="#1d4ed8"),
        yaxis="y2",
        customdata=freq_xlabels,
        hovertemplate="M0 txn=%{customdata}<br>M1 Retention=%{y:.1%}<extra></extra>",
    ))
    fig_combo_plain.add_hline(y=avg_ret, line_dash="dot", line_color="#f97316", line_width=1.5,
                              annotation_text=f"Avg Retention {avg_ret:.1%}",
                              annotation_position="right",
                              annotation_font_color="#f97316", yref="y2")
    fig_combo_plain.update_layout(
        title=dict(text=f"Biểu đồ phân phối số lượng KH theo số lượng giao dịch ở tháng đầu tiên và tỷ lệ quay lại ở tháng thứ 2<br><span style='font-size:12px;color:#6b7280;font-weight:normal'>Phân phối lệch phải - Median: {med_txn:.1f} giao dịch | Mean: {mean_txn:.1f} giao dịch</span>", font=dict(size=14, weight="bold"), x=0.5, xanchor="center"),
        height=390, margin=dict(l=40, r=70, t=90, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(size=11),
        yaxis=dict(title="Số user", showgrid=True, gridcolor="#f0f0f0",
                   zeroline=False, tickformat=","),
        yaxis2=dict(title="M1 Retention", overlaying="y", side="right",
                    tickformat=".0%", showgrid=False, zeroline=False,
                    range=[max(0, freq["ret"].min() - 0.08), min(1.0, freq["ret"].max() + 0.06)]),
        xaxis=dict(title="Số giao dịch M0", showgrid=False,
                   tickvals=list(freq.index), ticktext=freq_xlabels),
        barmode="relative",
    )
    # ── Phần 1: Validate ─────────────────────────────────────────────────────
    from scipy.stats import spearmanr as _spearmanr1c
    _rho_1c, _rho_p_1c = _spearmanr1c(m0_all["m0_txn"], m0_all["retained"])
    st.plotly_chart(fig_combo_plain, use_container_width=True, config={"displayModeBar": False})
    st.markdown("**1. Nhìn nhận chung về giả thuyết (Validation)**")
    st.markdown(f"Quan sát dữ liệu quá khứ, user thực hiện càng nhiều giao dịch trong tháng đầu tiên (M0) có **xu hướng** quay lại ở tháng tiếp theo (M1) cao hơn. Nhóm chỉ có 1 giao dịch có tỉ lệ retention thấp nhất ({freq.loc[1,'ret']:.1%}), thấp hơn trung bình retention tổng thể ({avg_ret:.1%}).")
    st.markdown(f"""Tuy nhiên, phân tích này tiếp tục thực hiện kiểm định thống kê: *Spearman correlation (one-tailed): kiểm định xem số giao dịch M0 có tương quan dương đơn điệu với tỷ lệ retention M1 không.*

- H₀: Không có mối tương quan dương giữa số giao dịch M0 và retention M1 (ρ ≤ 0)
- H₁: Số giao dịch M0 càng cao, tỷ lệ retention M1 càng cao (ρ > 0)

**Kết quả:** Spearman ρ = {_rho_1c:.3f}, p = {_rho_p_1c/2:.4f} (one-tailed) → Bác bỏ H₀ - có tương quan dương giữa số giao dịch M0 và retention M1.""")
    st.info(f"Giả thuyết được ủng hộ bởi dữ liệu xu hướng lịch sử và kiểm định thống kê - Spearman ρ = {_rho_1c:.3f} (p < 0.05). Đây là tương quan, chưa phải nhân quả.")

    st.markdown("---")

    # ── Phần 2: What is N? ────────────────────────────────────────────────────
    st.markdown("**2. What is N?**")
    st.markdown(f"<div style='text-align:center;font-size:16px;font-weight:700;margin-bottom:2px'>Phân phối số giao dịch M0 và tỷ lệ retention M1</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;color:#6b7280;font-size:13px'>Median: {med_txn:.1f} | Mean: {mean_txn:.1f} giao dịch. {pct_below:.1%} user tập trung ở vùng đỏ (&lt;{THRESHOLD} giao dịch).</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_combo, use_container_width=True, config={"displayModeBar": False})
    st.markdown("Quan sát dữ liệu lịch sử phát hiện điểm **bất thường** tại N=4. Khi thực hiện **kiểm tra đơn lẻ** ở **2024** và **2025** đều ghi nhận mức tăng **bất thường** tại 4 transaction. Phân tích tiếp tục dùng **Decision Tree** để xác định ngưỡng tách tối ưu một cách khách quan.")

    # Decision Tree: 3 cột nhỏ
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier as _DTC
    cohort_year_map = m0_all.merge(
        filt[["user_id","cohort"]].drop_duplicates(), on="user_id", how="left"
    )
    cohort_year_map["year"] = cohort_year_map["cohort"].apply(lambda x: int(str(x)[:4]))

    def _dt_result(data):
        if len(data) < 30 or data["retained"].nunique() < 2:
            return None
        try:
            dt = _DTC(max_depth=1, random_state=42)
            dt.fit(data[["m0_txn"]], data["retained"])
            thr = dt.tree_.threshold[0]
            n_split = int(thr) + 1
            hi = data[data["m0_txn"] >= n_split]["retained"].mean()
            lo = data[data["m0_txn"] < n_split]["retained"].mean()
            return thr, n_split, lo, hi, len(data)
        except Exception:
            return None

    _dt_all  = _dt_result(m0_all)
    _dt_2024 = _dt_result(cohort_year_map[cohort_year_map["year"]==2024])
    _dt_2025 = _dt_result(cohort_year_map[cohort_year_map["year"]==2025])

    def _dt_fig(thr, n_split, lo, hi, n_total, title, figsize=(3.2, 2.6), fs=8):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        ax.add_patch(plt.Rectangle((2.8, 3.8), 4.4, 1.5, fc="#dbeafe", ec="#93c5fd", lw=1.5, zorder=2))
        ax.text(5, 4.85, f"Số GD M0 ≤ {thr:.1f}", ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.text(5, 4.2, f"n = {n_total:,}", ha="center", va="center", fontsize=fs-1, color="#6b7280")
        ax.annotate("", xy=(2.2, 2.9), xytext=(4.0, 3.8), arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1))
        ax.annotate("", xy=(7.8, 2.9), xytext=(6.0, 3.8), arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1))
        ax.text(2.5, 3.5, "True", fontsize=fs-1, color="#6b7280")
        ax.text(6.8, 3.5, "False", fontsize=fs-1, color="#6b7280")
        ax.add_patch(plt.Rectangle((0.2, 1.2), 4.0, 1.5, fc="#fee2e2", ec="#fca5a5", lw=1.2, zorder=2))
        ax.text(2.2, 2.2, f"1–{n_split-1} GD", ha="center", va="center", fontsize=fs-0.5, fontweight="bold", color="#dc2626")
        ax.text(2.2, 1.6, f"Ret: {lo:.1%}", ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.add_patch(plt.Rectangle((5.8, 1.2), 4.0, 1.5, fc="#dcfce7", ec="#86efac", lw=1.2, zorder=2))
        ax.text(7.8, 2.2, f"≥{n_split} GD", ha="center", va="center", fontsize=fs-0.5, fontweight="bold", color="#15803d")
        ax.text(7.8, 1.6, f"Ret: {hi:.1%}", ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.set_title(title, fontsize=fs+1, fontweight="bold", pad=6)
        ax.text(5, 0.3, f"N gợi ý: ≥{n_split}", ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#1d4ed8",
                bbox=dict(fc="#eff6ff", ec="#bfdbfe", boxstyle="round,pad=0.3"))
        plt.tight_layout(pad=0.4)
        return fig

    # Chart tổng thể - to, ở giữa
    _dca, _dcb, _dcc = st.columns([1, 4, 1])
    with _dcb:
        if _dt_all:
            st.pyplot(_dt_fig(*_dt_all, "Tổng thể", figsize=(5.5, 3.4), fs=10), use_container_width=True)

    # 2 chart nhỏ - 2024 và 2025
    _dd1, _dd2 = st.columns(2)
    with _dd1:
        if _dt_2024:
            st.pyplot(_dt_fig(*_dt_2024, "Cohort 2024", figsize=(3.2, 2.6), fs=8), use_container_width=True)
        else:
            st.caption("Không đủ dữ liệu cho Cohort 2024")
    with _dd2:
        if _dt_2025:
            st.pyplot(_dt_fig(*_dt_2025, "Cohort 2025", figsize=(3.2, 2.6), fs=8), use_container_width=True)
        else:
            st.caption("Không đủ dữ liệu cho Cohort 2025")

    st.info(f"Cả quan sát và Decision Tree đều chỉ về N=3–4. Cohort 2024 (chiếm 76% data) xác nhận rõ N=4, cohort 2025 ra N=3 nhưng chênh lệch không đáng kể.\n\nCân nhắc chọn **N=4** là ngưỡng. Tuy nhiên khi **triển khai thực tế**, cần thực hiện **A/B Testing** để xác nhận.")

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.markdown(f"Câu hỏi tiếp theo: tại sao **N=4** lại là điểm có ý nghĩa, không phải N=3 hay N=5? **Phân tích GMV** và **cấu trúc phí** cho thấy cơ chế kinh tế đằng sau.")

    drv_plot = drv[drv.index <= 12].copy()
    drv_plot["fee_20k"] = 20_000 / drv_plot["med_gmv"]
    drv_plot["x"] = drv_plot.index.astype(str)
    colors_drv = ["#ef4444" if i < THRESHOLD else "#1d4ed8" for i in drv_plot.index]

    # Fee/GMV chart + retention line overlay
    _ret_drv = [float(freq.loc[i, "ret"]) if i in freq.index else float("nan") for i in drv_plot.index]
    fig_drv = go.Figure()
    fig_drv.add_trace(go.Bar(
        x=drv_plot["x"], y=drv_plot["fee_20k"],
        name="Tỷ lệ phí", yaxis="y",
        marker_color=colors_drv, opacity=0.85,
        text=[f"{v:.0%}" for v in drv_plot["fee_20k"]], textposition="outside",
        textfont=dict(size=9),
        hovertemplate="Txn=%{x}<br>Phí/tổng chi tiêu=%{y:.0%}<extra></extra>",
    ))
    fig_drv.add_trace(go.Scatter(
        x=drv_plot["x"], y=_ret_drv,
        name="Retention M1", yaxis="y2",
        mode="lines+markers+text",
        line=dict(color="#111827", width=2),
        marker=dict(size=6, color="#111827"),
        text=[f"{v:.0%}" if not (isinstance(v, float) and v != v) else "" for v in _ret_drv],
        textposition="top center",
        textfont=dict(size=9, color="#111827"),
        hovertemplate="Txn=%{x}<br>Retention M1=%{y:.1%}<extra></extra>",
    ))
    fig_drv = clean_fig(fig_drv, height=350)
    fig_drv.update_layout(
        xaxis=dict(title="Số giao dịch M0", showgrid=False),
        yaxis=dict(tickformat=".0%", title="Phí / Tổng chi tiêu", range=[0, drv_plot["fee_20k"].max()*1.35]),
        yaxis2=dict(tickformat=".0%", title="Retention M1", overlaying="y", side="right", range=[0.5, 1.0], showgrid=False),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08, font=dict(size=10)),
        margin=dict(t=40),
    )
    _drv_l, _drv_c, _drv_r = st.columns([1, 8, 1])
    with _drv_c:
        st.markdown("<div style='text-align:center;font-size:14px;font-weight:700;margin-bottom:0px'>Tỷ lệ phí và Retention M1 theo số giao dịch M0</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar": False})

    r1 = drv.loc[1] if 1 in drv.index else None
    r4 = drv.loc[4] if 4 in drv.index else None
    fee1 = f"{r1['med_fee_pct']:.0%}" if r1 is not None else "~15%"
    fee4 = f"{r4['med_fee_pct']:.0%}" if r4 is not None else "~2.7%"
    gmv1 = f"{r1['med_gmv']/1000:.0f}K" if r1 is not None else "133K"
    gmv4 = f"{r4['med_gmv']/1000:.0f}K" if r4 is not None else "747K"
    st.markdown(f"Phí 20K/tháng là **cố định** - nhóm 1 giao dịch (tổng chi tiêu {gmv1}) chịu tỷ lệ phí {fee1}, nhóm 4–6 giao dịch (tổng chi tiêu {gmv4}) chỉ còn {fee4}. **N=4 là điểm mà sản phẩm bắt đầu hợp lý về chi phí với người dùng.**")

    st.markdown("---")

    # ── Phần 4: Causation + A/B Test ─────────────────────────────────────────
    st.markdown("**4. Đánh giá và đề xuất thực hiện A/B testing**")
    st.markdown(f"""- **Tương quan ≠ nhân quả:** User quẹt nhiều trong M0 có thể đơn giản là vì họ có nhiều nhu cầu hơn - không phải vì số giao dịch khiến họ gắn bó. Tặng voucher để ép đủ 4 lần chưa chắc tạo ra thói quen thực sự khi hết khuyến mãi.
- **Đề xuất A/B Test:** Tặng voucher/hoàn tiền nhắm vào giao dịch thứ 3 và thứ 4 của user mới - đây là vùng mà mỗi giao dịch thêm cải thiện retention nhiều nhất, và cũng là ngưỡng phí/GMV bắt đầu hợp lý. Không nên dàn voucher từ giao dịch 1 (tốn nguồn lực, user có thể giữ lại ngay cả không cần) hoặc nhóm >7 lần (đã tự gắn bó). MDE = 5% → ~1,300 user/nhóm; acquisition hiện tại ~600 user/tháng: cần ~4 tháng để đủ mẫu.""")
    st.info(f"Giả thuyết được ủng hộ - users hoàn thành ≥{THRESHOLD} giao dịch trong M0 có retention M1 cao hơn có ý nghĩa thống kê (Spearman ρ = {_rho_1c:.3f}, p < 0.05). N = {THRESHOLD} là điểm bão hòa exploratory, cần A/B test để xác nhận nhân quả.")

# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Identify users showing signs of churn. How do they differ? What to recommend?")

    st.markdown("---")
    st.markdown("#### 1. Identify - Ai đang rời?")

    st.markdown("**Kể từ đầu 2025 đến tháng 6/2025, Quick Ratio của Buy Now Pay Later luôn nhỏ hơn 1 - tốc độ user rời đi nhiều hơn số user mới mỗi tháng.**")

    # ── Growth accounting ─────────────────────────────────────────────────────
    months_by_user = filt.groupby("user_id")["trxn_month"].apply(set).to_dict()
    all_um = filt.set_index(["user_id","trxn_month"])["month_number"].to_dict()
    rows_ga = []
    for (uid, month), mn in all_um.items():
        prev = month - 1
        if mn == 0:
            utype = "New"
        elif prev in months_by_user.get(uid, set()):
            utype = "Retained"
        else:
            utype = "Resurrected"
        rows_ga.append({"month": str(month), "user_type": utype})
    ga = pd.DataFrame(rows_ga).groupby(["month","user_type"]).size().unstack(fill_value=0)
    ga.index = pd.PeriodIndex(ga.index, freq="M")
    ga = ga.sort_index()
    for col in ["New","Retained","Resurrected"]:
        if col not in ga.columns: ga[col] = 0
    ga["MAU"] = ga[["New","Retained","Resurrected"]].sum(axis=1)
    ga_r = ga.reset_index()
    ga_r["churned"] = 0
    for i in range(1, len(ga_r)):
        prev_mau = ga_r.loc[i-1, "MAU"]
        retained = ga_r.loc[i, "Retained"]
        ga_r.loc[i, "churned"] = max(prev_mau - retained, 0)
    ga_r["quick_ratio"] = ga_r["New"] / ga_r["churned"].replace(0, np.nan)
    ga_r = ga_r[ga_r["churned"] > 0]
    x_ga = [str(m) for m in ga_r["month"]]

    qr_vals = ga_r["quick_ratio"].tolist()
    marker_colors = ["#1d4ed8" if v >= 1 else "#ef4444" for v in qr_vals]

    fig_qr = go.Figure()
    # Area traces - label mỗi 2 tháng để tránh chen
    _lbl_new  = [f"{v:,}" if i % 2 == 0 else "" for i, v in enumerate(ga_r["New"])]
    _lbl_churn = [f"{v:,}" if i % 2 == 1 else "" for i, v in enumerate(ga_r["churned"])]
    fig_qr.add_trace(go.Scatter(
        x=x_ga, y=ga_r["New"], name="New users",
        mode="markers+text", fill="tozeroy",
        marker=dict(size=5, color="rgba(59,130,246,0.5)", line=dict(width=0)),
        line=dict(color="rgba(59,130,246,0.65)", width=1.5, dash="dot"),
        fillcolor="rgba(59,130,246,0.07)", yaxis="y",
        text=_lbl_new, textposition="top center",
        textfont=dict(size=8, color="rgba(59,130,246,0.85)"),
        hovertemplate="%{x}<br>New: %{y:,}<extra></extra>",
    ))
    fig_qr.add_trace(go.Scatter(
        x=x_ga, y=ga_r["churned"], name="Churn User",
        mode="markers+text", fill="tozeroy",
        marker=dict(size=5, color="rgba(239,68,68,0.5)", line=dict(width=0)),
        line=dict(color="rgba(239,68,68,0.65)", width=1.5, dash="dot"),
        fillcolor="rgba(239,68,68,0.07)", yaxis="y",
        text=_lbl_churn, textposition="bottom center",
        textfont=dict(size=8, color="rgba(239,68,68,0.85)"),
        hovertemplate="%{x}<br>Churn User: %{y:,}<extra></extra>",
    ))
    # QR = 1 reference (trục phải)
    fig_qr.add_trace(go.Scatter(
        x=x_ga, y=[1]*len(x_ga), name="QR = 1",
        mode="lines", line=dict(color="#111827", width=1.5, dash="dot"),
        yaxis="y2", showlegend=True, hoverinfo="skip",
    ))
    # QR line - đen, marker xanh lá khi ≥1, đỏ khi <1
    _dot_colors = ["#16a34a" if v >= 1 else "#ef4444" for v in qr_vals]
    fig_qr.add_trace(go.Scatter(
        x=x_ga, y=qr_vals, name="Quick Ratio",
        mode="lines+markers", yaxis="y2",
        line=dict(color="#111827", width=2.5),
        marker=dict(size=8, color=_dot_colors, line=dict(width=1.5, color="white")),
        hovertemplate="%{x}<br><b>QR: %{y:.2f}</b><extra></extra>",
    ))

    fig_qr.update_layout(
        title=dict(text="Net Growth BNPL User",
                   font=dict(size=19, weight="bold"), x=0.5, xanchor="center"),
        height=420, margin=dict(l=20, r=50, t=75, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        barmode="overlay",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)),
        font=dict(size=13),
        xaxis=dict(showgrid=False, tickangle=0, tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Số user", font=dict(size=12)), tickfont=dict(size=11),
                   showgrid=True, gridcolor="#f0f0f0", zeroline=False, tickformat=","),
        yaxis2=dict(title=dict(text="Quick Ratio", font=dict(size=12)), tickfont=dict(size=11),
                    overlaying="y", side="right", showgrid=False, zeroline=False,
                    range=[0, max(qr_vals) * 1.15]),
        shapes=[dict(type="line", x0="2025-01", x1="2025-01", y0=0, y1=1, yref="paper",
                     line=dict(color="#6b7280", width=1, dash="dot"))],
        annotations=[dict(
            x=x_ga[-1], y=0.72, xref="x", yref="y2",
            text="QR < 1 | Tăng trưởng user âm",
            showarrow=False, font=dict(size=10, color="#111827"),
            xanchor="right", yanchor="top",
        )],
    )
    st.plotly_chart(fig_qr, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""Điều này đặt ra hai bài toán:
1. Tìm hiểu tại sao user rời bỏ sản phẩm BNPL
2. Phát triển đầu vào bằng cách thu hút khách hàng mới
""")
    st.markdown("""
<div style="background:#fef2f2;border-left:4px solid #ef4444;padding:12px 16px;border-radius:6px;margin-bottom:24px">
<span style="color:#374151">Trong tình huống này, đổ thêm ngân sách vào acquisition chỉ bù đắp tạm thời.</span><br>
<b style="color:#b91c1c">Tuy nhiên nếu không hiểu được lý do user rời đi, user mới vào rồi có thể cũng sẽ rời theo pattern cũ.</b>
</div>
<h4 style="margin-top:0">Churn Threshold</h4>
""", unsafe_allow_html=True)

    # Chart: phân phối khoảng cách giao dịch (tính trước để dùng trong text)
    _tmp = df[df["user_id"].isin(filt_users)][["user_id","trxn_date"]].sort_values(["user_id","trxn_date"])
    _tmp = _tmp.copy()
    _tmp["prev"] = _tmp.groupby("user_id")["trxn_date"].shift(1)
    _tmp["gap"] = (_tmp["trxn_date"] - _tmp["prev"]).dt.days
    _gaps = _tmp["gap"].dropna()
    _gaps = _gaps[(_gaps > 0) & (_gaps <= 35)]
    _med_gap = int(_gaps.median())
    _p95_gap = int(_gaps.quantile(0.95))

    st.markdown("Trong sản phẩm tài chính tín dụng, user không cần đóng tài khoản khi muốn ngừng dùng - cần xác định ngưỡng im lặng để gắn nhãn churn. Với BNPL, **30 ngày** là ngưỡng hợp lý dựa trên hai lý do:")
    st.markdown("**1. Billing cycle theo tháng, phí 20.000đ/kỳ** - mỗi kỳ là một lần user phải chủ động quyết định có dùng tiếp hay không.")
    st.markdown(f"**2. Hành vi giao dịch thực tế** - phân phối khoảng cách giữa 2 lần mua cho thấy P95 = {_p95_gap} ngày. Vượt 30 ngày là ngoài vùng hành vi bình thường (xem biểu đồ bên dưới).")

    _fig_gap = go.Figure()
    _fig_gap.add_trace(go.Histogram(
        x=_gaps, nbinsx=60, name="Số lượt",
        marker_color="#bfdbfe", opacity=0.85,
        hovertemplate="Khoảng cách: %{x} ngày<br>Số lượt: %{y:,}<extra></extra>",
    ))
    _fig_gap.add_vline(x=_med_gap, line_dash="solid", line_color="#1d4ed8", line_width=2,
                       annotation_text=f"Median {_med_gap} ngày",
                       annotation_position="top right", annotation_font_color="#1d4ed8")
    _fig_gap.add_vline(x=_p95_gap, line_dash="dash", line_color="#ef4444", line_width=2,
                       annotation_text=f"P95 = {_p95_gap} ngày",
                       annotation_position="top left", annotation_font_color="#ef4444")
    _fig_gap.update_layout(
        title=dict(text="Phân phối khoảng cách giữa 2 lần giao dịch liên tiếp",
                   font=dict(size=15), x=0.5, xanchor="center"),
        height=320, margin=dict(l=50, r=30, t=55, b=50),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False, font=dict(size=11),
        xaxis=dict(title="Số ngày", showgrid=False, range=[0, 35]),
        yaxis=dict(title="Số lượt", showgrid=True, gridcolor="#f0f0f0", tickformat=","),
    )
    _gL, _gC, _gR = st.columns([1, 6, 1])
    with _gC:
        st.plotly_chart(_fig_gap, use_container_width=True, config={"displayModeBar": False})
    st.info("**Churn = không phát sinh giao dịch trong 30 ngày liên tiếp.**\n\n*Lưu ý: với production data có daily granularity, ngưỡng này cần được re-validate.*")

    st.markdown("---")
    st.markdown("#### 2. How do they differ?")
    st.markdown("Nhìn vào M0 - tháng user phát sinh giao dịch mua đầu tiên - hai nhóm có profile gần giống nhau. Sự khác biệt nằm ở **tần suất**, **mức độ đa dạng dịch vụ** và **gánh nặng phí tương đối**.")
    grp_c = m0s_filt[~m0s_filt["retained"]]
    grp_a = m0s_filt[ m0s_filt["retained"]]
    _m5uid = set(active[active["month_number"] == 5]["user_id"]) & set(filt_users)
    grp_m5 = m0s_filt[m0s_filt.index.isin(_m5uid)]

    def _med(g, col): return g[col].median()
    def _avg(g, col): return g[col].mean()

    def _cell(med_str, avg_str):
        return f"<div style='font-size:17px;font-weight:700'>{med_str}</div><div style='font-size:11px;color:#9ca3af;margin-top:2px'>avg {avg_str}</div>"

    def _delta_badge(val_c, val_m5, unit="", k=False):
        diff = val_m5 - val_c
        if k: return f"+{diff/1000:.0f}K"
        if "%" in unit: return f"+{diff:.1%}"
        return f"+{diff:.0f}{unit}"

    _rows_data = [
        ("Số user", f"<b>{len(grp_c):,}</b>", f"<b>{len(grp_a):,}</b>", f"<b>{len(grp_m5):,}</b>"),
        ("Số giao dịch",
         _cell(f"{_med(grp_c,'n_txn'):.0f}", f"{_avg(grp_c,'n_txn'):.1f}"),
         _cell(f"{_med(grp_a,'n_txn'):.0f}", f"{_avg(grp_a,'n_txn'):.1f}"),
         _cell(f"{_med(grp_m5,'n_txn'):.0f}", f"{_avg(grp_m5,'n_txn'):.1f}")),
        ("Số loại dịch vụ",
         _cell(f"{_med(grp_c,'n_carriers'):.0f}", f"{_avg(grp_c,'n_carriers'):.1f}"),
         _cell(f"{_med(grp_a,'n_carriers'):.0f}", f"{_avg(grp_a,'n_carriers'):.1f}"),
         _cell(f"{_med(grp_m5,'n_carriers'):.0f}", f"{_avg(grp_m5,'n_carriers'):.1f}")),
        ("Tổng chi tiêu",
         _cell(f"{_med(grp_c,'gmv')/1000:.0f}K", f"{_avg(grp_c,'gmv')/1000:.0f}K"),
         _cell(f"{_med(grp_a,'gmv')/1000:.0f}K", f"{_avg(grp_a,'gmv')/1000:.0f}K"),
         _cell(f"{_med(grp_m5,'gmv')/1000:.0f}K", f"{_avg(grp_m5,'gmv')/1000:.0f}K")),
        ("Giá trị mỗi GD",
         _cell(f"{_med(grp_c,'med_aov')/1000:.0f}K", f"{_avg(grp_c,'med_aov')/1000:.0f}K"),
         _cell(f"{_med(grp_a,'med_aov')/1000:.0f}K", f"{_avg(grp_a,'med_aov')/1000:.0f}K"),
         _cell(f"{_med(grp_m5,'med_aov')/1000:.0f}K", f"{_avg(grp_m5,'med_aov')/1000:.0f}K")),
        ("Phí / Tổng chi tiêu",
         _cell(f"{_med(grp_c,'fee_pct'):.1%}", f"{_avg(grp_c,'fee_pct'):.1%}"),
         _cell(f"{_med(grp_a,'fee_pct'):.1%}", f"{_avg(grp_a,'fee_pct'):.1%}"),
         _cell(f"{_med(grp_m5,'fee_pct'):.1%}", f"{_avg(grp_m5,'fee_pct'):.1%}")),
    ]
    _tbl_html = """<table style='width:100%;border-collapse:collapse;font-size:14px'>
<tr>
  <th style='text-align:left;padding:10px 12px;font-weight:600;width:28%;color:#374151;background:#f9fafb'><div style='font-size:10px;color:#9ca3af;font-weight:400;margin-top:2px'>trung vị / avg</div></th>
  <th style='text-align:center;padding:10px 12px;font-weight:700;color:#fff;background:#ef4444'>Churn @ M1</th>
  <th style='text-align:center;padding:10px 12px;font-weight:700;color:#fff;background:#1d4ed8'>Active M1</th>
  <th style='text-align:center;padding:10px 12px;font-weight:700;color:#fff;background:#059669'>Active M5</th>
</tr>"""
    _tag_equal = "<span style='font-size:10px;background:#fef9c3;color:#92400e;padding:1px 6px;border-radius:3px;margin-left:6px;font-weight:500'>≈ bằng nhau</span>"
    for row in _rows_data:
        _lbl = row[0]; _vc = row[1]; _va = row[2]; _vm5 = row[3]
        _is_user = (_lbl == "Số user")
        _is_aov  = (_lbl == "Giá trị mỗi GD")
        _lbl_style = "padding:12px;color:#6b7280;font-style:italic;vertical-align:middle;background:#f9fafb" if _is_user else "padding:12px;color:#374151;vertical-align:middle;background:#f9fafb"
        _lbl_html  = _lbl + (_tag_equal if _is_aov else "")
        _cc = ("#6b7280","#6b7280","#6b7280") if _is_user else ("#ef4444","#1d4ed8","#059669")
        _tbl_html += f"""<tr style='border-bottom:1px solid #f3f4f6'>
  <td style='{_lbl_style}'>{_lbl_html}</td>
  <td style='text-align:center;padding:12px;color:{_cc[0]};vertical-align:middle;background:#fff5f5'>{_vc}</td>
  <td style='text-align:center;padding:12px;color:{_cc[1]};vertical-align:middle;background:#eff6ff'>{_va}</td>
  <td style='text-align:center;padding:12px;color:{_cc[2]};vertical-align:middle;background:#f0fdf4'>{_vm5}</td>
</tr>"""
    _tbl_html += "</table>"
    st.markdown(_tbl_html, unsafe_allow_html=True)

    st.caption(f"So sánh profile M0 của 3 nhóm: Churn @ M1 ({len(grp_c):,} users) / Active M1 ({len(grp_a):,}) / Active M5 ({len(grp_m5):,}). Nhóm active càng lâu có xu hướng giao dịch nhiều hơn và phí tương đối thấp hơn ngay từ M0.")

    st.markdown("---")
    st.markdown("##### Hành vi sử dụng trong Tháng đầu tiên (M0)")

    # Carrier Diversity
    ct_div = m0s_filt.groupby("n_carriers").agg(
        n=("retained","count"), churn=("retained", lambda x: 1-x.mean()), med_gmv=("gmv","median")
    ).query("n>=100").reset_index()
    avg_ch = 1 - m0s_filt["retained"].mean()
    colors_div = ["#ef4444" if c > avg_ch else "#6b7280" for c in ct_div["churn"]]
    fig2 = go.Figure(go.Bar(
        x=ct_div["n_carriers"], y=ct_div["churn"],
        marker_color=colors_div, width=0.5,
        text=[f"{c:.1%}" for c in ct_div["churn"]], textposition="outside",
        customdata=np.stack([ct_div["n"], ct_div["med_gmv"]/1000], axis=-1),
        hovertemplate="Churn: %{text}<br>n=%{customdata[0]:,}<br>Med GMV: %{customdata[1]:.0f}K<extra></extra>",
    ))
    fig2.add_hline(y=avg_ch, line_dash="dot", line_color="#9ca3af",
                   annotation_text=f"Avg {avg_ch:.1%}", annotation_position="bottom left")
    fig2 = clean_fig(fig2, height=260)
    fig2.update_layout(
        title=dict(text="Churn theo số loại dịch vụ M0", font=dict(size=12), x=0.5, xanchor="center"),
        xaxis=dict(title="Số loại dịch vụ",
                   tickvals=ct_div["n_carriers"],
                   ticktext=[f"{int(n)} loại" for n in ct_div["n_carriers"]]),
        yaxis=dict(tickformat=".0%"),
    )
    _cd1 = ct_div[ct_div["n_carriers"]==1]
    _cd2 = ct_div[ct_div["n_carriers"]==2]
    _churn1 = f"{_cd1['churn'].values[0]:.1%}" if len(_cd1) else "31.8%"
    _gmv1   = f"{_cd1['med_gmv'].values[0]/1000:.0f}K" if len(_cd1) else "197K"
    _gmv2   = f"{_cd2['med_gmv'].values[0]/1000:.0f}K" if len(_cd2) else "478K"
    _, _p_div, *_ = chi2_contingency(pd.crosstab(m0s_filt["n_carriers"], m0s_filt["retained"]))

    # n_txn chart
    _txn_tmp = m0s_filt.copy()
    _txn_tmp["txn_grp"] = pd.cut(_txn_tmp["n_txn"], bins=[0,1,3,9999],
                                  labels=["1 lần","2–3 lần","4+ lần"])
    _ct_txn_grp = _txn_tmp.groupby("txn_grp", observed=True).agg(
        n=("retained","count"), churn=("retained", lambda x: 1-x.mean())
    ).reset_index()
    _txn_colors = ["#ef4444" if c > avg_ch else "#6b7280" for c in _ct_txn_grp["churn"]]
    fig_txn = go.Figure(go.Bar(
        x=_ct_txn_grp["txn_grp"].astype(str), y=_ct_txn_grp["churn"],
        marker_color=_txn_colors, width=0.5,
        text=[f"{c:.1%}" for c in _ct_txn_grp["churn"]], textposition="outside",
        hovertemplate="%{x}<br>Churn: %{y:.1%}<extra></extra>", showlegend=False,
    ))
    fig_txn.add_hline(y=avg_ch, line_dash="dot", line_color="#9ca3af",
                      annotation_text=f"Avg {avg_ch:.1%}", annotation_position="top right")
    fig_txn = clean_fig(fig_txn, height=260)
    fig_txn.update_layout(
        title=dict(text="Churn theo số giao dịch M0", font=dict(size=12), x=0.5, xanchor="center"),
        xaxis_title="Số giao dịch",
        yaxis=dict(tickformat=".0%"),
    )
    _, _p_txn, *_ = chi2_contingency(pd.crosstab(
        pd.cut(m0s_filt["n_txn"], bins=[0,1,3,9999], labels=["1","2-3","4+"]),
        m0s_filt["retained"]
    ))

    _div_col, _txn_col = st.columns(2)
    with _div_col:
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"48% user dùng 1 loại dịch vụ - churn {_churn1}, trên TB {avg_ch:.1%}. Dùng ≥2 loại, GMV tăng từ {_gmv1} → {_gmv2}, churn giảm dưới mức TB. (chi-square p={_p_div:.4f})")
    with _txn_col:
        st.plotly_chart(fig_txn, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Churn giảm rõ rệt khi đạt ≥4 giao dịch trong M0 - ngưỡng bắt đầu hình thành thói quen. (chi-square p={_p_txn:.4f})")

    st.markdown("**Churn Rate: Số giao dịch × Số loại dịch vụ** (đỏ = trên mức TB, xanh = dưới mức TB)")

    _hm = m0s_filt.copy()
    _txn_labels = ["1","2","3","4","5-6","7-9","10+"]
    _hm["txn_bin"] = pd.cut(_hm["n_txn"], bins=[0,1,2,3,4,6,9,float("inf")],
                            labels=_txn_labels)
    _hm["txn_bin"] = pd.Categorical(_hm["txn_bin"].astype(str), categories=_txn_labels, ordered=True)
    _hm_grp = _hm.groupby(["n_carriers","txn_bin"], observed=True).agg(
        n=("retained","count"), churn=("retained", lambda x: 1-x.mean())
    ).reset_index()
    _hm_grp = _hm_grp[_hm_grp["n"] >= 30]
    _hm_pivot = _hm_grp.pivot(index="n_carriers", columns="txn_bin", values="churn")
    _hm_n     = _hm_grp.pivot(index="n_carriers", columns="txn_bin", values="n")
    _xs = [str(c) for c in _hm_pivot.columns]
    _ys = [f"{int(r)} loại" for r in _hm_pivot.index]
    _zs = _hm_pivot.values.tolist()
    _texts = []
    for r in _hm_pivot.index:
        _row = []
        for c in _hm_pivot.columns:
            v = _hm_pivot.loc[r, c]
            n_val = _hm_n.loc[r, c] if (r in _hm_n.index and c in _hm_n.columns) else float("nan")
            _row.append("" if pd.isna(v) else f"{v:.0%}  n={int(n_val):,}")
        _texts.append(_row)
    _zflat = _hm_pivot.values[~np.isnan(_hm_pivot.values)]
    fig_hm = go.Figure(go.Heatmap(
        z=_zs, x=_xs, y=_ys, text=_texts, texttemplate="%{text}",
        colorscale=[[0,"#1d4ed8"],[0.5,"#f9fafb"],[1,"#c53030"]],
        zmid=float(_zflat.mean()) if len(_zflat) else 0.26,
        zmin=float(_zflat.min()) if len(_zflat) else 0.1,
        zmax=float(_zflat.max()) if len(_zflat) else 0.4,
        colorbar=dict(title="Churn rate", tickformat=".0%", thickness=14),
        hovertemplate="Churn: %{z:.1%}<extra></extra>",
    ))
    fig_hm = clean_fig(fig_hm, height=280)
    fig_hm.update_layout(
        xaxis=dict(title="Số giao dịch M0", type="category", categoryorder="array", categoryarray=_xs),
        yaxis=dict(title="Số loại dịch vụ", autorange="reversed"),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Ô trống = dưới 30 user.")
    _mat_avg = float(_zflat.mean()) if len(_zflat) else float("nan")
    st.markdown(f"Nhóm nguy cơ cao nhất: **1 giao dịch × 1 loại dịch vụ** - churn {_hm_pivot.iloc[0, 0]:.0%}, n={int(_hm_n.iloc[0,0]):,} user. Tăng cả tần suất lẫn mức độ đa dạng dịch vụ mới thực sự cải thiện retention - một trong hai không đủ.")

    st.markdown("---")
    st.markdown("##### Tác động của phí lên hành vi người dùng")
    st.markdown("#### Churn Rate by GMV (Gross Merchandise Value)")

    _m = m0s_filt.copy()
    _m["gmv_tier"] = pd.cut(_m["gmv"],
        bins=[0,200_000,400_000,700_000,float("inf")],
        labels=["<200K","200K–400K","400K–700K",">700K"])
    ts = _m.groupby("gmv_tier", observed=True).agg(
        n=("retained","count"), m1=("retained","mean"), med=("gmv","median")
    ).reset_index()
    ts["churn"] = 1 - ts["m1"]
    ts["fee_pct"] = 20_000 / ts["med"]
    _, p_chi, *_ = chi2_contingency(pd.crosstab(_m["gmv_tier"], _m["retained"]))

    fig3 = go.Figure(go.Bar(
        x=ts["gmv_tier"].astype(str), y=ts["churn"],
        marker_color=["#ef4444","#f97316","#6b7280","#1d4ed8"],
        text=[f"{v:.1%}" for v in ts["churn"]], textposition="outside", width=0.5,
    ))
    fig3 = clean_fig(fig3, height=260)
    fig3.update_layout(yaxis=dict(tickformat=".0%", range=[0, ts["churn"].max()*1.3]),
                       xaxis_title="GMV M0 tier")

    st.markdown(f"""
**Churn giảm đều theo mức chi tiêu:** nhóm GMV <200K churn {ts['churn'].iloc[0]:.1%}, nhóm >700K chỉ churn {ts['churn'].iloc[-1]:.1%}.

Phí 20K cố định tạo gánh nặng không đồng đều - nhóm chi tiêu thấp chịu tỷ lệ phí/GMV cao hơn gấp nhiều lần.
""")
    _f3l, _f3c, _f3r = st.columns([1, 6, 1])
    with _f3c:
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""**Kiểm định (chi-square):**

*Chi-square test of independence (two-tailed): kiểm định xem mức GMV trong M0 và tình trạng churn có liên hệ với nhau không.*

H₀: Tỷ lệ churn không khác nhau giữa các GMV tier

H₁: Tỷ lệ churn khác nhau giữa các GMV tier

**Kết quả:** p = {p_chi:.4f} → Bác bỏ H₀ - tỷ lệ churn khác nhau có ý nghĩa thống kê giữa các GMV tier. Quan sát từ dữ liệu: churn giảm dần theo GMV.""")

    st.markdown("---")
    st.markdown("#### Fee burden by GMV (Gross Merchandise Value)")
    st.markdown("Phí 20K cố định mỗi kỳ tạo gánh nặng không đồng đều theo mức chi tiêu. Câu hỏi: nhóm churn có chịu fee/GMV cao hơn nhóm còn lại không?")

    # Mann-Whitney U: fee_pct của churned vs retained
    _fee_clean = m0s_filt[m0s_filt["fee_pct"] <= 2]
    _fee_c = _fee_clean[_fee_clean["retained"]==False]["fee_pct"]
    _fee_r = _fee_clean[_fee_clean["retained"]==True]["fee_pct"]
    _mw_stat, _mw_p = mannwhitneyu(_fee_c, _fee_r, alternative="greater")
    from scipy.stats import spearmanr as _spearmanr
    _rho, _rho_p = _spearmanr(_fee_clean["fee_pct"], _fee_clean["retained"])

    _fee_buck = _fee_clean.copy()
    _fee_buck = _fee_buck[_fee_buck["fee_pct"] <= 0.30]
    _fee_buck["fee_bin"] = pd.cut(_fee_buck["fee_pct"],
        bins=[0, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30],
        labels=["<3%", "3-5%", "5-8%", "8-12%", "12-20%", "20-30%"])
    _fee_grp = _fee_buck.groupby("fee_bin", observed=True).agg(
        n=("retained","count"), churn=("retained", lambda x: 1-x.mean())
    ).reset_index()
    _fee_colors = ["#ef4444" if c > (1-m0s_filt["retained"].mean()) else "#6b7280"
                   for c in _fee_grp["churn"]]

    fig_fee = go.Figure(go.Bar(
        x=_fee_grp["fee_bin"].astype(str), y=_fee_grp["churn"],
        marker_color=_fee_colors, width=0.6,
        text=[f"{v:.0%}" for v in _fee_grp["churn"]], textposition="outside",
        customdata=_fee_grp["n"],
        hovertemplate="Fee/GMV: %{x}<br>Churn: %{y:.1%}<br>n=%{customdata:,}<extra></extra>",
    ))
    fig_fee = clean_fig(fig_fee, height=300)
    _avg_churn_fee = 1 - m0s_filt["retained"].mean()
    fig_fee.add_hline(y=_avg_churn_fee, line_dash="dot", line_color="#9ca3af",
                      annotation_text=f"Avg {_avg_churn_fee:.0%}", annotation_position="top right")
    fig_fee.update_layout(
        title=dict(text="Churn Rate theo Fee/GMV bucket", font=dict(size=15), x=0.5, xanchor="center"),
        xaxis=dict(title="Fee / GMV (M0)"),
        yaxis=dict(tickformat=".0%", title="Churn rate", range=[0, _fee_grp["churn"].max()*1.3]),
    )
    _ffL, _ffC, _ffR = st.columns([1, 6, 1])
    with _ffC:
        st.plotly_chart(fig_fee, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
**Kiểm định:**

*Mann-Whitney U (one-tailed): so sánh phân phối fee/GMV giữa 2 nhóm.*

H₀: Phân phối fee/GMV của nhóm churn bằng nhóm còn lại

H₁: Nhóm churn có fee/GMV cao hơn nhóm còn lại

**Kết quả:** p = {_mw_p:.4f} → Bác bỏ H₀ - nhóm churn có fee/GMV cao hơn nhóm còn lại có ý nghĩa thống kê.

*Spearman correlation: tương quan đơn điệu giữa fee/GMV và churn.*

**Kết quả:** Spearman ρ = {_rho:.3f}, p = {_rho_p:.4f} → Bác bỏ H₀ - tương quan dương giữa fee/GMV và churn (effect size nhỏ).

*Lưu ý: fee/GMV = 20K/GMV, không tách được effect phí vs chi tiêu.*
""")

    st.info("""**How do they differ?**
- Nhóm churn: ít loại dịch vụ (48% dùng 1 loại), ít giao dịch, GMV thấp → phí 20K chiếm tỷ lệ lớn
- AOV hai nhóm bằng nhau → không phải vấn đề ticket size, mà là tần suất và mức độ đa dạng dịch vụ
- Churn rate có liên hệ thống kê với cả carrier diversity (p < 0.05), GMV tier (p < 0.05), và fee/GMV (p < 0.05)""")

    st.markdown("---")
    st.markdown("#### 3. What to recommend?")

    st.markdown("**R1 - Kích hoạt khám phá dịch vụ trong tháng đầu** *(ngắn hạn)*")
    st.write("48% user chỉ dùng 1 loại dịch vụ trong M0, trung vị GMV 197K - phí 20K chiếm 10% chi tiêu. Nhóm dùng từ 2 loại trở lên có trung vị GMV 478K, cùng mức phí nhưng chỉ còn 4%. Đề xuất: trong 2 tuần đầu, gợi ý thử thêm 1 dịch vụ mới kèm cashback lần đầu ở category đó.")
    st.markdown("**R2 - Xem lại cấu trúc phí cho nhóm chi tiêu thấp** *(cần finance xác nhận)*")
    st.write("Cùng mức phí 20K cố định: nhóm GMV <200K chịu effective fee 17%, nhóm >700K chỉ 2% - chênh lệch gần 9 lần. Đề xuất: giữ 20K nhưng hoàn một phần phí cho user có GMV tháng dưới 500K.")
    st.markdown("**R3 - Theo dõi sát nhóm rủi ro cao**")
    st.write("Khoảng 1,084 user (6%) chỉ dùng Game/VNG và giao dịch ≤ 2 lần trong M0, churn rate 35.4% - cao hơn mức trung bình. Đây là nhóm cụ thể, dễ xác định, nên ưu tiên tiếp cận trước khi R1 được triển khai rộng.")
    st.info("R1 nhắm vào 48% user mới - dễ triển khai, không cần thay sản phẩm. R2 cần finance xác nhận trước. R3 có danh sách user cụ thể, có thể bắt đầu ngay.")

# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("So What? - For a Business Lead")

    # ── Shared computation ────────────────────────────────────────────────────
    _p3_total   = len(m0s_filt)
    _p3_churn_r = 1 - m0s_filt["retained"].sum() / _p3_total if _p3_total > 0 else 0
    _p3_ret_lo  = m0s_filt[m0s_filt["gmv"] < 200_000]["retained"].mean()
    _p3_ret_hi  = m0s_filt[m0s_filt["gmv"] > 700_000]["retained"].mean()

    _gmv_bins   = [0, 100_000, 200_000, 300_000, 500_000, 700_000, 1_000_000, float("inf")]
    _gmv_labels = ["Dưới 100K", "100K – 200K", "200K – 300K", "300K – 500K", "500K – 700K", "700K – 1M", "Trên 1M"]
    _ts3 = m0s_filt.copy()
    _ts3["gmv_tier"] = pd.cut(_ts3["gmv"], bins=_gmv_bins, labels=_gmv_labels)
    _ts3_grp = _ts3.groupby("gmv_tier", observed=True).agg(
        n=("gmv","count"), med_gmv=("gmv","median"), ret=("retained","mean"),
    ).reset_index()
    _ts3_grp["fee_pct"] = 20_000 / _ts3_grp["med_gmv"]

    _p3_new_mon  = filt[filt["month_number"]==0].groupby("trxn_month")["user_id"].nunique()
    _p3_new_peak = int(_p3_new_mon.max()) if len(_p3_new_mon) else 0
    _p3_new_last = int(_p3_new_mon.iloc[-1]) if len(_p3_new_mon) else 0
    _p3_new_drop = (_p3_new_last - _p3_new_peak) / _p3_new_peak if _p3_new_peak else float("nan")
    _p3_mau_mon  = filt.groupby("trxn_month")["user_id"].nunique()
    _p3_h24 = [p for p in _p3_mau_mon.index if str(p)[:4] == "2024" and 1 <= int(str(p)[5:7]) <= 6]
    _p3_h25 = [p for p in _p3_mau_mon.index if str(p)[:4] == "2025" and 1 <= int(str(p)[5:7]) <= 6]
    _p3_mau_24 = float(_p3_mau_mon[_p3_h24].mean()) if _p3_h24 else float("nan")
    _p3_mau_25 = float(_p3_mau_mon[_p3_h25].mean()) if _p3_h25 else float("nan")
    _p3_mau_yoy = (_p3_mau_25 - _p3_mau_24) / _p3_mau_24 if _p3_mau_24 else float("nan")

    # ── 1. What is happening ──────────────────────────────────────────────────
    st.markdown("#### 1. What is happening?")
    _p3_yoy_str = f"{_p3_mau_yoy:.0%}" if not np.isnan(_p3_mau_yoy) else "–"
    st.markdown(f"""
<div style="background:#fef2f2;border-left:4px solid #ef4444;padding:14px 18px;border-radius:6px">
<b style="color:#b91c1c">Trong 6 tháng đầu năm 2025, sản phẩm đang mất user nhanh hơn tốc độ có thể bù đắp, giảm {_p3_yoy_str} active user so với cùng kỳ 2024.</b>
</div>
""", unsafe_allow_html=True)
    st.markdown("")

    _p3_new_drop_str = f"{_p3_new_drop:+.0%} từ đỉnh {_p3_new_peak:,}" if not np.isnan(_p3_new_drop) else "–"
    def _kpi_card(label, value, delta, up=False):
        bg = "#f0fdf4" if up else "#fef2f2"
        fg = "#16a34a" if up else "#ef4444"
        return f"""<div style='text-align:center;padding:16px 8px'>
  <div style='font-size:14px;color:#374151;margin-bottom:6px'>{label}</div>
  <div style='font-size:32px;font-weight:500;color:#111827;line-height:1.1'>{value}</div>
  <div style='margin-top:8px;display:flex;justify-content:center'>
    <span style='font-size:12px;background:{bg};color:{fg};padding:3px 10px;border-radius:12px'>{delta}</span>
  </div>
</div>"""
    _mp, _c1, _c2, _c3, _mp2 = st.columns([0.5, 3, 3, 3, 0.5])
    _c1.markdown(_kpi_card("M1 churn rate", f"{_p3_churn_r:.1%}", "↑ Cứ 4 user mới, mất 1"), unsafe_allow_html=True)
    _c2.markdown(_kpi_card("Retention nhóm GMV thấp", f"{_p3_ret_lo:.1%}", f"↓ {(_p3_ret_lo-_p3_ret_hi):.1%} vs nhóm cao"), unsafe_allow_html=True)
    _c3.markdown(_kpi_card("New user/tháng (gần nhất)", f"{_p3_new_last:,}", f"↓ {_p3_new_drop_str}"), unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Why it happened ────────────────────────────────────────────────────
    st.markdown("#### 2. Why it happened?")
    st.markdown("##### Tỷ lệ rời bỏ cao chủ yếu đến từ việc người dùng chưa hình thành thói quen sử dụng sản phẩm ngay trong tháng đầu tiên (M0).")

    # Bar chart: churn rate by số loại dịch vụ + số giao dịch, highlight nhóm thấp nhất
    _hb = m0s_filt.copy()
    _hb["n_cat_lbl"] = _hb["n_carriers"].clip(upper=4).apply(lambda x: f"{x} loại" if x<4 else "4+ loại")
    _hb["n_txn_lbl"] = pd.cut(_hb["n_txn"], bins=[0,1,3,9999],
                               labels=["1 lần","2–3 lần","4+ lần"]).astype(str)
    _hb_cat = _hb.groupby("n_cat_lbl")["retained"].agg(churn=lambda x: 1-x.mean(), n="count").reset_index()
    _hb_txn = _hb.groupby("n_txn_lbl")["retained"].agg(churn=lambda x: 1-x.mean(), n="count").reset_index()
    _avg_churn = 1 - m0s_filt["retained"].mean()

    _cat_order = [f"{i} loại" for i in range(1,4)] + ["4+ loại"]
    _hb_cat = _hb_cat[_hb_cat["n_cat_lbl"].isin(_cat_order)].set_index("n_cat_lbl").reindex(_cat_order).dropna().reset_index()
    _txn_order = ["1 lần","2–3 lần","4+ lần"]
    _hb_txn = _hb_txn.set_index("n_txn_lbl").reindex(_txn_order).dropna().reset_index()

    _hcol1, _hcol2 = st.columns(2)
    for _col, _df, _xlbl, _title in [
        (_hcol1, _hb_cat, "n_cat_lbl", "Churn rate theo số loại dịch vụ M0"),
        (_hcol2, _hb_txn, "n_txn_lbl", "Churn rate theo số giao dịch M0"),
    ]:
        _colors = ["#ef4444" if i==0 else "#6b7280" for i in range(len(_df))]
        _fig_h = go.Figure()
        _fig_h.add_trace(go.Bar(
            x=_df[_xlbl], y=_df["churn"], marker_color=_colors,
            text=[f"{v:.1%}" for v in _df["churn"]], textposition="outside",
            hovertemplate="%{x}<br>Churn: %{y:.1%}<extra></extra>", showlegend=False,
        ))
        _fig_h.add_hline(y=_avg_churn, line_dash="dot", line_color="#9ca3af", line_width=1.5,
                         annotation_text=f"Avg {_avg_churn:.1%}", annotation_position="right",
                         annotation_font_color="#9ca3af", annotation_font_size=10)
        _fig_h.update_layout(
            title=dict(text=_title, font=dict(size=11), x=0.5, xanchor="center"),
            height=250, margin=dict(l=30,r=50,t=35,b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#f0f0f0",
                       range=[0, _df["churn"].max()*1.3]),
        )
        with _col:
            st.plotly_chart(_fig_h, use_container_width=True, config={"displayModeBar":False})

    st.markdown(f"""
- **Đa dạng danh mục - Điểm chạm giữ chân:** Nhóm chỉ dùng 1 loại dịch vụ có churn {_hb_cat.iloc[0]['churn']:.1%} (trên mức trung bình). Khi chuyển sang ≥2 loại, churn giảm rõ rệt - trải nghiệm đa dịch vụ tạo stickiness với hệ sinh thái.
- **Tần suất - Magic Number:** Churn giảm mạnh khi user đạt ≥4 lần giao dịch trong M0. Đây là ngưỡng bắt đầu hình thành thói quen.

##### Phí 20K khuếch đại vấn đề ở nhóm chi tiêu thấp.
""")


    _tbl = _ts3_grp.copy()
    _tbl["Chi tiêu tháng đầu"] = _tbl["gmv_tier"].astype(str)
    _tbl["Số user"]             = _tbl["n"]
    _tbl["Phí 20K / chi tiêu"] = _tbl["fee_pct"]
    _tbl["Retention tháng 2"]  = _tbl["ret"]
    _tbl["Churn rate"]          = 1 - _tbl["ret"]

    def _bg_red(s):
        lo, hi = s.min(), s.max()
        return [f"background-color:rgba(239,68,68,{0.08+0.72*(v-lo)/(hi-lo+1e-9):.2f})" for v in s]
    def _bg_green(s):
        lo, hi = s.min(), s.max()
        return [f"background-color:rgba(34,197,94,{0.08+0.72*(v-lo)/(hi-lo+1e-9):.2f})" for v in s]

    _styled = (
        _tbl[["Chi tiêu tháng đầu","Số user","Phí 20K / chi tiêu","Retention tháng 2","Churn rate"]].style
        .apply(_bg_red,   subset=["Phí 20K / chi tiêu","Churn rate"])
        .apply(_bg_green, subset=["Retention tháng 2"])
        .format({"Số user":"{:,}", "Phí 20K / chi tiêu":"{:.1%}", "Retention tháng 2":"{:.1%}", "Churn rate":"{:.1%}"})
        .set_properties(subset=["Số user","Phí 20K / chi tiêu","Retention tháng 2","Churn rate"], **{"text-align": "center"})
    )
    st.markdown("<style>div[data-testid='stDataFrame'] th,div[data-testid='stDataFrame'] td{text-align:center!important}</style>", unsafe_allow_html=True)
    st.dataframe(_styled, hide_index=True, use_container_width=True)
    _churn_low  = 1 - float(_ts3_grp.iloc[0]["ret"])  if len(_ts3_grp) else float("nan")
    _fee_low    = float(_ts3_grp.iloc[0]["fee_pct"])   if len(_ts3_grp) else float("nan")
    _churn_high = 1 - float(_ts3_grp.iloc[-1]["ret"])  if len(_ts3_grp) else float("nan")
    st.markdown(f"""
- Nhóm chi tiêu thấp nhất (dưới 100K): phí 20K chiếm {_fee_low:.0%} tổng chi tiêu, churn {_churn_low:.1%} - cao nhất trong tất cả các nhóm.
- Nhóm chi tiêu cao (trên 500K): tỷ lệ phí/chi tiêu xuống dưới 4%, churn {_churn_high:.1%}. Phí không còn là yếu tố quyết định với nhóm này.
- Ngay cả nhóm chi tiêu cao nhất vẫn churn {_churn_high:.1%} - cho thấy có một phần churn tự nhiên không liên quan đến phí, cần tìm hiểu thêm qua định tính.
""")

    st.markdown("---")

    # ── 3. What should we do next ─────────────────────────────────────────────
    st.markdown("#### 3. What should we do next?")
    st.markdown("""
**Cross-sell & Onboarding:** Thiết kế các campaign/nhiệm vụ thúc đẩy người dùng mới trải nghiệm ít nhất 2 loại dịch vụ khác nhau và đạt tối thiểu 4 lần giao dịch trong 30 ngày đầu tiên.

**Cải tiến chính sách phí:**
- Cân nhắc miễn phí 20K cho tháng đầu tiên (M0) để người dùng tập trung trải nghiệm dịch vụ và hình thành thói quen mà không gặp rào cản tài chính.
- Hoặc áp dụng chính sách phí lũy tiến/hoàn phí (ví dụ: miễn phí hoặc hoàn lại 20K nếu chi tiêu đạt trên một hạn mức nhất định hoặc số lần giao dịch đạt ngưỡng thói quen).
""")
    st.info("**Người dùng rời đi vì chưa hình thành thói quen trước khi bị tính phí. Xây dựng hành vi chi tiêu lặp lại trong M0 (nhiệm vụ hằng ngày, chiến dịch, game,...) là phương pháp có thể cân nhắc để cải thiện retention.**")
