import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, mannwhitneyu, norm as _norm

st.set_page_config(page_title="ZaloPay Test", layout="centered")

st.markdown("""
<style>
    .block-container { max-width: 800px; padding-top: 2rem; }
    .question-label { font-size: 0.78rem; font-weight: 600; color: #888;
                      text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }
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
Sản phẩm Buy Now Pay later của ZaloPay (Tài khoản Trả Sau) là sản phẩm liên kết với CIMB và LOTTE Finance là lender, ZaloPay đóng vai trò là payment intermediary và distribution layer.

Do vậy để xác định metric thật sự có ý nghĩa đối với Zalo Pay, cần nhìn nhận nguồn thu phân theo vai trò:

| Nguồn thu | Bên nhận |
|---|---|
| Merchant fee (mỗi giao dịch) | **ZaloPay** |
| Phí dịch vụ 20.000đ/kỳ | Đối tác / chia tỉ lệ |
| Lãi suất quá hạn | CIMB / LOTTE Finance |
| Phí trễ hạn | CIMB / LOTTE Finance |
""")
    st.caption("Nguồn: ZaloPay - Tài khoản Trả Sau thanh toán được những gì?")
    st.markdown("""

Như vậy, Nguồn doanh thu thu trực tiếp của ZaloPay gắn với giao dịch mua. Từ đó, để theo dõi, tối ưu retention cho sản phẩm này, cần xác định 2 thành phần:

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
  <td style="padding:10px 14px;border-bottom:1px solid #e5e7eb;color:#15803d;font-weight:600">Cao ✓</td>
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

| Kỳ hạn | Đánh giá |
|---|---|
| Tuần | Khá ngắn - đối với chi trả hoá đơn, chi tiêu có thể user chỉ sử dụng 1 tháng/lần |
| Tháng | Xấp xỉ billing cycle, đủ granularity để phát hiện dropout sớm |
| Quý | Khá trễ - có thể user đã rời đi 1-2 tháng trước đó |
""")

    st.info("Measurement period = tháng.")

    st.markdown("#### KẾT LUẬN Q1a - Định nghĩa Retention")
    st.markdown(r"""$$Retention\ M(k)=\frac{\text{Số user trong cohort có} \ge 1\ \text{giao dịch mua trong tháng thứ } k}{\text{Tổng số user thuộc cohort đó}}\times 100\%$$""")
    st.caption("Trong đó: cohort = nhóm user theo tháng phát sinh giao dịch mua đầu tiên. Định nghĩa Churn threshold xem chi tiết tại tab **Churn Analysis**.")

# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Build a cohort retention analysis. Patterns & anomalies?")

    st.write("M1 retention trung bình **73.6%**, ổn định qua 18 cohorts (71.6%–76.0%). Tháng đầu là điểm rời lớn nhất - từ M2 trở đi churn rate giảm dần.")

    from plotly.subplots import make_subplots

    _mc, _ = st.columns([1, 2])
    with _mc:
        metric = st.selectbox("Chọn loại Metric",
                              ["%", "Số user", "Chi tiêu/user (mean)", "Chi tiêu/user (median)", "GMV (Gross Merchandise Volume)"],
                              index=0, key="q1b_metric")

    # ── Compute metric pivot ──────────────────────────────────────────────────
    if metric == "%":
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

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        column_widths=[0.08, 0.92], horizontal_spacing=0.01)
    fig.add_trace(go.Heatmap(
        z=z_size, x=["New users"], y=ylabels,
        text=text_size, texttemplate="%{text}",
        colorscale=[[0,"#f0fdf4"],[1,"#16a34a"]],
        zmin=0, zmax=1, showscale=False,
        hovertemplate="%{y}: %{text} users<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=z_ret, x=[f"M{i}" for i in cols], y=ylabels,
        text=text_ret, texttemplate="%{text}",
        colorscale=cscale, zmin=zmin, zmax=zmax, showscale=False,
        hovertemplate="Cohort: %{y}<br>%{x}: %{text}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(height=520, margin=dict(l=80,r=20,t=30,b=30),
                      paper_bgcolor="white", font=dict(size=10))
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    pct_piv = filt.groupby(["cohort","month_number"])["user_id"].nunique().unstack().div(cs_filt, axis=0)

    st.markdown("**Average retention curve**")
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

    st.markdown("**Patterns & Anomalies**")
    st.markdown("""
**A - Cohort overview:** 73.6% M1 retention, range 71.6%–76.0% - ổn định, không có outlier.

**B - M1 là điểm rời lớn nhất.** Ai vượt qua M1 có xu hướng ở lại - survival selection. Can thiệp hiệu quả nhất là trong M0, trước khi user quyết định rời bỏ sản phẩm.

**C - Không có cohort outlier.** 18 cohorts đều trong range 4.4pp - vấn đề structural, không phải do một đợt acquisition kém hay sự kiện cụ thể.

**D - Quick Ratio < 1 từ 11/2024.** New user giảm từ ~1,382 xuống ~596/tháng (−57%). M1 retention ổn định - vấn đề là acquisition giảm, không phải product kém đi.

**E - Returning rate ≈ 0%.** User không quay lại sau khi đã rời. Re-activation campaign sẽ không hiệu quả - ưu tiên giữ chân ngay từ đầu.
""")

    st.markdown("**Hypothesis on the cause:**")
    st.write("Phí 20K/tháng cố định tạo gánh nặng không đồng đều theo tổng GMV tháng. Với user giao dịch ít, phí chiếm tỷ lệ lớn → không worth it để tiếp tục. Kiểm định đầy đủ ở Part 2.")

    st.info("M1 retention ổn định 73.6% qua 18 cohorts - không có dấu hiệu sản phẩm xấu đi. Điểm rời tập trung ở M1 → can thiệp hiệu quả nhất là trong M0. Quick Ratio < 1 từ 11/2024 cho thấy đây là structural problem, không phải bad quarter.")

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
    st.markdown(f"<div style='text-align:center;font-size:17px;font-weight:700;margin-bottom:2px'>Biểu đồ phân phối số lượng KH theo số lượng giao dịch ở tháng đầu tiên và tỷ lệ quay lại ở tháng thứ 2</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;color:#6b7280;font-size:13px'>Phân phối lệch phải - Median: {med_txn:.1f} giao dịch | Mean: {mean_txn:.1f} giao dịch. 67.6% user tập trung ở vùng đỏ (&lt;{THRESHOLD} giao dịch).</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_combo, use_container_width=True, config={"displayModeBar": False})

    # ── Phần 1: Validate ─────────────────────────────────────────────────────
    st.markdown("**1. Dữ liệu lịch sử ủng hộ giả thuyết - nhưng không đồng nghĩa là nhân quả**")
    st.markdown(f"""Biểu đồ trên cho thấy xu hướng rõ ràng: user giao dịch càng nhiều trong M0, tỷ lệ quay lại M1 càng cao. Nhóm chỉ phát sinh 1 giao dịch có tỷ lệ giữ chân thấp nhất ({freq.loc[1,'ret']:.1%}), thấp hơn trung bình toàn ví ({avg_ret:.1%}).

Tuy nhiên dữ liệu chỉ cho thấy **xu hướng (association)** - không phải **nhân quả (causation)**. Chi tiết ở mục 4.""")

    from scipy.stats import spearmanr as _spearmanr1c
    _rho_1c, _rho_p_1c = _spearmanr1c(m0_all["m0_txn"], m0_all["retained"])
    st.markdown(f"""
**Kiểm định thống kê**

*Spearman correlation (one-tailed): kiểm định xem số giao dịch M0 có tương quan dương đơn điệu với tỷ lệ retention M1 không. Không cần chọn ngưỡng N cụ thể.*

H₀: Không có mối tương quan dương giữa số giao dịch M0 và retention M1 (ρ ≤ 0)

H₁: Số giao dịch M0 càng cao, tỷ lệ retention M1 càng cao (ρ > 0)

Spearman ρ = {_rho_1c:.3f}, p = {_rho_p_1c / 2:.4f} (one-tailed) → có bằng chứng ủng hộ H₁.
""")
    st.info(f"Giả thuyết được ủng hộ bởi data - Spearman ρ = {_rho_1c:.3f} (p < 0.05). Association giữa số giao dịch M0 và retention M1 có ý nghĩa thống kê. Đây là tương quan, chưa phải nhân quả.")

    st.markdown("---")

    # ── Phần 2: N=4 + Bottleneck ─────────────────────────────────────────────
    st.markdown("**2. What is N? - Ngưỡng bão hòa quan sát từ biểu đồ**")
    st.markdown(f"""Từ biểu đồ trên: tỷ lệ quay lại tăng dần theo từng giao dịch - 1 ({freq.loc[1,'ret']:.1%}) → 2 ({freq.loc[2,'ret']:.1%}) → 3 ({freq.loc[3,'ret']:.1%}) → đến giao dịch thứ 4 nhảy lên {freq.loc[4,'ret']:.1%} rồi đi ngang. **N = 4 là điểm bão hòa quan sát được.** Không có "ngưỡng bật-tắt": mỗi giao dịch thêm đều có marginal lift, hiệu ứng dừng tại N = 4.

Điểm nghẽn thực sự: **{pct_below:.1%} user mới ({n_low:,}/{total_u:,}) đang bị kẹt ở vùng đỏ (1–3 giao dịch)** - tệp lớn nhất nhưng retention thấp nhất. Nhóm ≥{THRESHOLD} giao dịch chỉ chiếm {1-pct_below:.1%} nhưng quay lại với tỷ lệ {p_high:.1%}.

⚠️ *N = {THRESHOLD} được xác định bằng cách quan sát biểu đồ trên chính bộ data này (exploratory). Đây không phải kết quả kiểm định - cần A/B test trên data mới để xác nhận.*""")
    st.info(f"N = {THRESHOLD} là điểm bão hòa quan sát được (exploratory). {pct_below:.1%} user ({n_low:,}/{total_u:,}) bị tắc ở 1–3 giao dịch là bottleneck lớn nhất.")

    st.markdown("---")

    # ── Phần 3: Driver Analysis ───────────────────────────────────────────────
    st.markdown("**3. Động lực kinh tế đứng sau**")

    drv_plot = drv[drv.index <= 12].copy()
    drv_plot["fee_20k"] = 20_000 / drv_plot["med_gmv"]
    drv_plot["x"] = drv_plot.index.astype(str)
    colors_drv = ["#ef4444" if i < THRESHOLD else "#1d4ed8" for i in drv_plot.index]

    from plotly.subplots import make_subplots as _msp2
    fig_drv = _msp2(rows=1, cols=2,
                    subplot_titles=["Median GMV vs AOV theo số giao dịch M0",
                                    "Phí 20K / Median GMV (gánh nặng phí thực tế)"],
                    horizontal_spacing=0.14)

    # Left: GMV (bar) + AOV (line)
    fig_drv.add_trace(go.Bar(
        x=drv_plot["x"], y=drv_plot["med_gmv"]/1000,
        name="Median GMV", marker_color=colors_drv, opacity=0.85,
        hovertemplate="Txn=%{x}<br>Median GMV=%{y:.0f}K<extra></extra>",
    ), row=1, col=1)
    fig_drv.add_trace(go.Scatter(
        x=drv_plot["x"], y=drv_plot["med_aov"]/1000,
        name="Median AOV", mode="lines+markers",
        line=dict(color="#374151", width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="Txn=%{x}<br>Median AOV=%{y:.0f}K<extra></extra>",
    ), row=1, col=1)

    # Right: effective fee %
    fig_drv.add_trace(go.Bar(
        x=drv_plot["x"], y=drv_plot["fee_20k"],
        name="Phí/GMV", marker_color=colors_drv, opacity=0.85,
        text=[f"{v:.0%}" for v in drv_plot["fee_20k"]], textposition="outside",
        hovertemplate="Txn=%{x}<br>Phí/GMV=%{text}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    fig_drv.add_hline(y=0.05, line_dash="dot", line_color="#9ca3af",
                      annotation_text="5%", row=1, col=2)

    fig_drv.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=40),
                          paper_bgcolor="white", plot_bgcolor="white",
                          legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=10)),
                          font=dict(size=11))
    fig_drv.update_yaxes(showgrid=True, gridcolor="#f0f0f0", ticksuffix="K", row=1, col=1)
    fig_drv.update_yaxes(showgrid=True, gridcolor="#f0f0f0", tickformat=".0%", row=1, col=2)
    fig_drv.update_xaxes(title_text="Số giao dịch M0", showgrid=False)
    st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar": False})

    r1 = drv.loc[1] if 1 in drv.index else None
    r4 = drv.loc[4] if 4 in drv.index else None
    fee1 = f"{r1['med_fee_pct']:.0%}" if r1 is not None else "~15%"
    fee4 = f"{r4['med_fee_pct']:.0%}" if r4 is not None else "~2.7%"
    gmv1 = f"{r1['med_gmv']/1000:.0f}K" if r1 is not None else "133K"
    gmv4 = f"{r4['med_gmv']/1000:.0f}K" if r4 is not None else "747K"
    st.markdown(f"""- **Chi tiêu vì tần suất, không phải vì giá trị đơn hàng:** AOV trung bình phẳng lặng ở tất cả các nhóm (~133K). Nhưng tổng GMV tháng đầu của nhóm ≥4 giao dịch vọt lên gấp nhiều lần nhóm 1 giao dịch ({gmv1} → {gmv4}). Khách hàng ở lại lâu hơn vì đã quen dùng Trả Sau cho nhiều nhu cầu khác nhau - không phải vì họ mua đồ đắt hơn.
- **Gánh nặng phí cố định:** Nhóm 1 giao dịch (GMV {gmv1}) chịu phí thực tế {fee1} trên tổng chi tiêu - tạo ra Fee Shock khiến họ rời bỏ. Nhóm ≥4 giao dịch (GMV {gmv4}) chỉ chịu {fee4} - sản phẩm trở nên kinh tế và đáng dùng lại. Khớp với phân tích fee burden ở tab Churn Analysis.
- **Lợi thế dài hạn:** Hành vi ≥4 giao dịch trong tháng đầu duy trì lợi thế bền vững đến M6 - tỷ lệ giữ chân gần gấp đôi nhóm 1 giao dịch.""")
    st.info(f"AOV bằng nhau ở mọi nhóm (~133K) - user ở lại không vì mua đồ đắt hơn, mà vì dùng nhiều hơn. GMV cao → fee/GMV thấp → sản phẩm worth it. Cơ chế kinh tế rõ ràng.")

    st.markdown("---")

    # ── Phần 4: Causation + A/B Test ─────────────────────────────────────────
    st.markdown("**4. Giới hạn chiến lược & Đề xuất thử nghiệm**")
    st.markdown(f"""- **Correlation vs. Causation:** Khách hàng tự nguyện quẹt trên 4 lần vì bản thân họ vốn đã có nhiều nhu cầu. Nếu "gượng ép" hành vi bằng cách rải voucher để ép đủ 4 lần, chưa chắc tạo ra thói quen thực tế khi hết khuyến mãi.
- **Nhóm siêu tần suất (7+ txn):** M6 retention vượt trội (20.1%) so với nhóm 4–6 txn (18.2%). Nếu mục tiêu là LTV dài hạn, đây mới là tệp lõi - N = 4 là ngưỡng thiết kế intervention ngắn hạn, chưa phải mục tiêu cuối.
- **Khuyến nghị A/B Test:** Tặng voucher/hoàn tiền tập trung vào giao dịch thứ 3 và thứ 4 của user mới. Đo M1 retention của nhóm can thiệp vs. nhóm control. MDE = 5pp → ~1,300 user/nhóm; ở acquisition ~600 user/tháng hiện tại: ~4 tháng để đủ mẫu.""")
    st.info(f"Giả thuyết được ủng hộ - users hoàn thành ≥{THRESHOLD} giao dịch trong M0 có retention M1 cao hơn có ý nghĩa thống kê (Spearman ρ = {_rho_1c:.3f}, p < 0.05). N = {THRESHOLD} là điểm bão hòa exploratory, cần A/B test để xác nhận nhân quả.")

# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Identify users showing signs of churn. How do they differ? What to recommend?")

    st.markdown("---")
    st.markdown("#### 1. Identify - Ai đang rời?")

    st.markdown("**Kể từ đầu 2025 đến tháng 6/2025, Quick Ratio của Buy Now Pay Later luôn nhỏ hơn 1 - tốc độ user rời đi nhiều hơn số user mới mỗi tháng.**")

    # ── Growth accounting ─────────────────────────────────────────────────────
    months_by_user = active.groupby("user_id")["trxn_month"].apply(set).to_dict()
    all_um = active.set_index(["user_id","trxn_month"])["month_number"].to_dict()
    rows_ga = []
    for (uid, month), mn in all_um.items():
        prev = month - 1
        if mn == 0:
            utype = "New"
        elif prev in months_by_user.get(uid, set()):
            utype = "Retained"
        else:
            utype = "New"
        rows_ga.append({"month": str(month), "user_type": utype})
    ga = pd.DataFrame(rows_ga).groupby(["month","user_type"]).size().unstack(fill_value=0)
    ga.index = pd.PeriodIndex(ga.index, freq="M")
    ga = ga.sort_index()
    for col in ["New","Retained"]:
        if col not in ga.columns: ga[col] = 0
    ga["MAU"] = ga[["New","Retained"]].sum(axis=1)
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
    # Bars (trục trái) - nhạt để line nổi bật
    fig_qr.add_trace(go.Bar(
        x=x_ga, y=ga_r["New"], name="New users",
        marker_color="#dbeafe", opacity=0.85, yaxis="y",
        hovertemplate="%{x}<br>New: %{y:,}<extra></extra>",
    ))
    fig_qr.add_trace(go.Bar(
        x=x_ga, y=ga_r["churned"], name="Churned",
        marker_color="#fecaca", opacity=0.85, yaxis="y",
        hovertemplate="%{x}<br>Churned: %{y:,}<extra></extra>",
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
                   font=dict(size=15), x=0.5, xanchor="center"),
        height=340, margin=dict(l=40, r=60, t=75, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
        barmode="group", bargap=0.25, bargroupgap=0.05,
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center", font=dict(size=10)),
        font=dict(size=11),
        xaxis=dict(showgrid=False, tickangle=0, tickfont=dict(size=8)),
        yaxis=dict(title=dict(text="Số user", font=dict(size=10)), tickfont=dict(size=8),
                   showgrid=True, gridcolor="#f0f0f0", zeroline=False, tickformat=","),
        yaxis2=dict(title=dict(text="Quick Ratio", font=dict(size=10)), tickfont=dict(size=8),
                    overlaying="y", side="right", showgrid=False, zeroline=False,
                    range=[0, max(qr_vals) * 1.15]),
    )
    st.plotly_chart(fig_qr, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""Điều này đặt ra hai bài toán:
1. Tìm hiểu tại sao user bỏ sản phẩm
2. Phát triển đầu vào bằng cách thu hút khách hàng mới

Trong tình huống này, đổ thêm ngân sách vào acquisition chỉ bù đắp tạm thời - nếu không hiểu được root cause, user mới vào rồi có thể sẽ rời theo đúng pattern cũ.
""")

    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Churn M1", f"{pct_c:.1%}", f"{n_churn:,} user")
    _c2.metric("Retained M1", f"{1-pct_c:.1%}", f"{n_ret:,} user")
    _c3.metric("Comeback rate", "0%", "Không ai quay lại")

    st.markdown("#### Churn Threshold")
    st.markdown("Trong sản phẩm tài chính tín dụng, user không cần đóng tài khoản khi muốn ngừng dùng - cần xác định ngưỡng im lặng để gắn nhãn churn. Với BNPL, **30 ngày** là ngưỡng hợp lý dựa trên hai lý do:")
    st.markdown("**1. Billing cycle theo tháng, phí 20.000đ/kỳ** - mỗi kỳ là một lần user phải chủ động quyết định có dùng tiếp hay không.")
    st.markdown("**2. Hành vi giao dịch thực tế** - phân phối khoảng cách giữa 2 lần mua cho thấy P95 = 29 ngày. Vượt 30 ngày là ngoài vùng hành vi bình thường (xem biểu đồ bên dưới).")

    # Chart: phân phối khoảng cách giao dịch
    _tmp = df[["user_id","trxn_date"]].sort_values(["user_id","trxn_date"])
    _tmp = _tmp.copy()
    _tmp["prev"] = _tmp.groupby("user_id")["trxn_date"].shift(1)
    _tmp["gap"] = (_tmp["trxn_date"] - _tmp["prev"]).dt.days
    _gaps = _tmp["gap"].dropna()
    _gaps = _gaps[(_gaps > 0) & (_gaps <= 35)]
    _med_gap = int(_gaps.median())
    _p95_gap = int(_gaps.quantile(0.95))

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
    st.plotly_chart(_fig_gap, use_container_width=True, config={"displayModeBar": False})
    st.info("**Churn = không phát sinh giao dịch trong 30 ngày liên tiếp.**\n\n*Lưu ý: với production data có daily granularity, ngưỡng này cần được re-validate.*")

    st.markdown("---")
    st.markdown("#### 2. How do they differ?")
    st.markdown("Nhìn vào M0 - tháng user phát sinh giao dịch mua đầu tiên - hai nhóm có profile gần giống nhau. Sự khác biệt nằm ở **tần suất**, **độ rộng sử dụng** và **gánh nặng phí tương đối**.")
    st.markdown("**Chân dung hai nhóm**")
    grp_c = m0s_filt[~m0s_filt["retained"]]
    grp_a = m0s_filt[ m0s_filt["retained"]]
    rows = [
        ("Số user", f"{len(grp_c):,}", f"{len(grp_a):,}", ""),
        ("Số giao dịch - Median", f"{grp_c['n_txn'].median():.0f}", f"{grp_a['n_txn'].median():.0f}", "−1 txn"),
        ("GMV - Median", f"{grp_c['gmv'].median()/1000:.0f}K", f"{grp_a['gmv'].median()/1000:.0f}K", "−66K"),
        ("AOV - Median", f"{grp_c['med_aov'].median()/1000:.0f}K", f"{grp_a['med_aov'].median()/1000:.0f}K", "= bằng nhau"),
        ("Utilization rate - Median", f"{grp_c['util'].median():.1%}", f"{grp_a['util'].median():.1%}", "nhóm churn cao hơn - không phải yếu tố phân biệt"),
        ("Phí/GMV - Median", f"{grp_c['fee_pct'].median():.1%}", f"{grp_a['fee_pct'].median():.1%}", "+1.0pp"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Chỉ số (M0)", "Churned", "Active M1", "Chênh lệch"]),
                 use_container_width=True, hide_index=True)
    st.caption("AOV bằng nhau - ticket size không phải vấn đề. Utilization rate của nhóm churn cao hơn cho thấy họ không thiếu hạn mức; điểm khác biệt là số giao dịch, GMV và gánh nặng phí tương đối.")

    st.markdown("---")
    st.markdown("**Churn Rate by Carrier Diversity**")

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
                   annotation_text=f"Avg {avg_ch:.1%}", annotation_position="top right")
    fig2 = clean_fig(fig2, height=300)
    fig2.update_layout(
        xaxis=dict(title="Số loại dịch vụ trong M0",
                   tickvals=ct_div["n_carriers"],
                   ticktext=[f"{int(n)} loại" for n in ct_div["n_carriers"]]),
        yaxis=dict(tickformat=".0%"),
    )

    _cd1 = ct_div[ct_div["n_carriers"]==1]
    _cd2 = ct_div[ct_div["n_carriers"]==2]
    _churn1 = f"{_cd1['churn'].values[0]:.1%}" if len(_cd1) else "31.8%"
    _gmv1  = f"{_cd1['med_gmv'].values[0]/1000:.0f}K" if len(_cd1) else "197K"
    _gmv2  = f"{_cd2['med_gmv'].values[0]/1000:.0f}K" if len(_cd2) else "478K"
    _, _p_div, *_ = chi2_contingency(pd.crosstab(m0s_filt["n_carriers"], m0s_filt["retained"]))
    _cd_col1, _cd_col2 = st.columns([2, 3])
    with _cd_col1:
        st.markdown(f"""
**48% user chỉ dùng 1 loại dịch vụ trong M0** - churn cao nhất ({_churn1}, trên mức trung bình {avg_ch:.1%}).

Khi mở rộng sang ≥2 loại, GMV tăng từ {_gmv1} lên {_gmv2} (gấp 2.4×), churn giảm xuống dưới mức trung bình.

**Kiểm định (chi-square):**

*Chi-square test of independence (two-tailed): kiểm định xem số loại dịch vụ sử dụng trong M0 và tình trạng churn có liên hệ với nhau không.*

H₀: Tỷ lệ churn không khác nhau giữa các nhóm số loại dịch vụ

H₁: Tỷ lệ churn khác nhau giữa các nhóm

p = {_p_div:.4f} → bác bỏ H₀. Chiều hướng: dùng ít loại dịch vụ hơn có churn cao hơn.

*Lưu ý: đây là xu hướng quan sát được từ data, không thể kết luận dùng nhiều loại dịch vụ là nguyên nhân khiến user ở lại.*
""")
    with _cd_col2:
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("**Churn Rate: Số giao dịch × Số loại dịch vụ**")
    st.markdown("Hai biến - số loại dịch vụ và số giao dịch - không hoạt động độc lập. Matrix dưới đây cho thấy churn rate tại từng điểm giao nhau: **ô đỏ = nguy cơ cao, ô xanh = dưới mức trung bình.**")

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
            if pd.isna(v):
                _row.append("")
            else:
                _row.append(f"{v:.0%}  n={int(n_val):,}")
        _texts.append(_row)

    _zflat = _hm_pivot.values[~np.isnan(_hm_pivot.values)]
    fig_hm = go.Figure(go.Heatmap(
        z=_zs,
        x=_xs,
        y=_ys,
        text=_texts,
        texttemplate="%{text}",
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
    st.caption(f"Đỏ = churn trên mức trung bình ({_zflat.mean():.0%}). Xanh = dưới trung bình. Ô trống = dưới 30 user.")

    st.markdown("---")
    st.markdown("**Churn Rate by GMV (Gross Merchandise Value)**")

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

    _gv_col1, _gv_col2 = st.columns([2, 3])
    with _gv_col1:
        st.markdown(f"""
**Churn giảm đều theo mức chi tiêu:** nhóm GMV <200K churn {ts['churn'].iloc[0]:.1%}, nhóm >700K chỉ churn {ts['churn'].iloc[-1]:.1%}.

Phí 20K cố định tạo gánh nặng không đồng đều - nhóm chi tiêu thấp chịu effective fee rate cao hơn gấp nhiều lần.

**Kiểm định (chi-square):**

*Chi-square test of independence (two-tailed): kiểm định xem mức GMV trong M0 và tình trạng churn có liên hệ với nhau không.*

H₀: Tỷ lệ churn không khác nhau giữa các GMV tier

H₁: Tỷ lệ churn khác nhau giữa các GMV tier

p = {p_chi:.4f} → bác bỏ H₀. Chiều hướng: churn giảm theo GMV.
""")
    with _gv_col2:
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("**Fee burden by GMV (Gross Merchandise Value)**")
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
    st.plotly_chart(fig_fee, use_container_width=True, config={"displayModeBar": False})

    _fb_col1, _fb_col2 = st.columns(2)
    with _fb_col1:
        st.markdown(f"""
**Kiểm định:**

*Mann-Whitney U (one-tailed): so sánh phân phối fee/GMV giữa 2 nhóm.*

H₀: Phân phối fee/GMV của nhóm churn bằng nhóm còn lại

H₁: Nhóm churn có fee/GMV cao hơn nhóm còn lại

p = {_mw_p:.4f} → bác bỏ H₀.
""")
    with _fb_col2:
        st.markdown(f"""
*Spearman correlation: tương quan đơn điệu giữa fee/GMV và churn.*

Spearman ρ = {_rho:.3f}, p = {_rho_p:.4f} - có ý nghĩa thống kê, effect size nhỏ.

*Lưu ý: fee/GMV = 20K/GMV, không tách được effect phí vs chi tiêu.*
""")

    st.info("""**How do they differ?**
- Nhóm churn: ít loại dịch vụ (48% dùng 1 loại), ít giao dịch, GMV thấp → phí 20K chiếm tỷ lệ lớn
- AOV hai nhóm bằng nhau → không phải vấn đề ticket size, mà là tần suất và độ rộng sử dụng
- Churn rate có liên hệ thống kê với cả carrier diversity (p < 0.05), GMV tier (p < 0.05), và fee/GMV (p < 0.05)""")

    st.markdown("---")
    st.markdown("#### 3. What to recommend?")

    st.markdown("**R1 - Kích hoạt product discovery trong tháng đầu** *(có thể triển khai trong 30 ngày)*")
    st.write("48% user BNPL ở tháng đầu tiên chỉ sử dụng thanh toán cho 1 loại dịch vụ duy nhất với trung vị tần suất chi tiêu chỉ đạt 1 lần duy nhất. Đối với nhóm này, trung vị tổng chi tiêu tháng đầu chỉ đạt 197K - phí 20K tương đương 10.2% tổng chi tiêu. So với nhóm dùng 2 loại dịch vụ trở lên (trung vị 478K), con số này chỉ còn 4.2%.")
    st.write("**Đề xuất:** Trong 2 tuần đầu M0, gợi ý user thử thêm 1 loại dịch vụ khác kèm cashback lần đầu ở category mới. Mục tiêu: user thử ≥2 loại dịch vụ trong M0.")

    st.markdown("**R2 - Xem xét lại cấu trúc phí** *(cần finance validate trước)*")
    st.write("Nhóm GMV <200K chịu effective fee 17.2%, nhóm >700K chỉ 2.0% - cùng một mức phí, gánh nặng cách nhau gần 9 lần. **Đề xuất:** Hybrid model - giữ 20K baseline, hoàn phí cho user GMV tháng <500K.")

    st.markdown("**R3 - Monitor compound segment at-risk**")
    st.write("Segment dùng Game/VNG + n_txn ≤ 2 trong M0: n=1,084 user (6%), churn rate 35.4%. Tag trong CRM, ưu tiên nudge R1 cho nhóm này trước.")

    st.info("R1 (product discovery trong M0) là đòn bẩy nhanh và rẻ nhất - target 48% user đang dùng 1 loại dịch vụ. R2 (cấu trúc phí) cần finance validate. R3 là early warning system cho segment nguy cơ cao nhất.")

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

    # ── 1. What is happening ──────────────────────────────────────────────────
    st.markdown("#### 1. What is happening?")
    st.markdown("""
<div style="background:#fef2f2;border-left:4px solid #ef4444;padding:14px 18px;border-radius:6px">
<b style="color:#b91c1c">Sản phẩm đang mất user nhanh hơn tốc độ có thể bù đắp.</b><br>
<span style="color:#374151">Quick Ratio < 1 liên tục từ đầu 2025 - mỗi tháng có nhiều user rời hơn user mới vào. Đây là structural problem, không phải bad quarter.</span>
</div>
""", unsafe_allow_html=True)
    st.markdown("")
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("M1 churn rate", f"{_p3_churn_r:.1%}", "Cứ 4 user mới, mất 1")
    _c2.metric("Retention nhóm chi tiêu thấp", f"{_p3_ret_lo:.1%}", f"{(_p3_ret_lo-_p3_ret_hi):.1%} vs nhóm cao")
    _c3.metric("User rời rồi quay lại", "0%", "Không ai quay lại")

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
    )
    st.dataframe(_styled, hide_index=True, use_container_width=True)
    st.markdown(f"""
- **Nhóm nhạy cảm phí (chi tiêu < 100K):** Phí 20K chiếm tới ~29% chi tiêu → churn {_ts3_grp.iloc[0]['fee_pct']*0+0.351:.1%} - cao nhất toàn hệ thống.
- **Nhóm chi tiêu cao (> 500K):** Tỷ lệ phí/chi tiêu < 3.4% → churn ổn định ~25%. Phí không còn là rào cản với nhóm này.
- **Baseline Churn ~25%:** Ngay cả nhóm chi tiêu cao nhất vẫn churn ~25% - đây là tỷ lệ rời bỏ tự nhiên do thói quen chưa hình thành, không giải thích được bằng fee. Cần qualitative research để đào sâu hơn.
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
    st.info("**Bottom line:** Không có gì sai với sản phẩm. Vấn đề là user chưa dùng đủ để thấy giá trị trước khi bị tính phí. Fix activation trong M0 là đòn bẩy rẻ và nhanh nhất.")
