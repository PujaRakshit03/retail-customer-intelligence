from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from retail_intelligence.data import (  # noqa: E402
    audit_transactions,
    data_quality_summary,
    load_transactions,
    prepare_transactions,
)
from retail_intelligence.analytics import (  # noqa: E402
    customer_concentration,
    group_performance,
    overview_metrics,
    sales_over_time,
    customer_cohort_retention,
    temporal_repeat_purchase_validation,
    purchase_interval_summary,
    top_product_pairs,
    segment_future_outcomes,
    cancellation_summary,
)
from retail_intelligence.customers import (  # noqa: E402
    add_opportunity_flags,
    assign_customer_segments,
    summarize_customer_segments,
    build_rfm,
)

st.set_page_config(page_title="Retail Customer Intelligence", page_icon="🛍️", layout="wide")
st.title("Retail Customer Intelligence")
st.caption("Evidence-backed retail analytics and customer opportunity discovery")

DEFAULT_PATH = ROOT / "data" / "raw" / "Online Retail.xlsx"

@st.cache_data(show_spinner="Loading and validating transactions…")
def load_from_path(path_string: str, modified_time: float):
    raw = load_transactions(path_string)
    return raw, prepare_transactions(raw), audit_transactions(raw)

@st.cache_data(show_spinner="Preparing uploaded transactions…")
def load_uploaded_file(file_name: str, file_bytes: bytes):
    suffix = Path(file_name).suffix.lower()
    temp_path = ROOT / "data" / "processed" / f"uploaded_source{suffix}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(file_bytes)
    try:
        raw = load_transactions(temp_path)
        return raw, prepare_transactions(raw), audit_transactions(raw)
    finally:
        temp_path.unlink(missing_ok=True)

with st.sidebar:
    st.header("Data source")
    use_upload = st.toggle("Upload a dataset", value=False)
    uploaded = st.file_uploader("UCI Online Retail file", type=["xlsx", "xls", "csv"]) if use_upload else None
    st.caption("The app does not upload your data to an external service. Uploaded files are processed locally by this app and the temporary copy is removed after loading.")

try:
    if uploaded is not None:
        raw, prepared, audit = load_uploaded_file(uploaded.name, uploaded.getvalue())
    else:
        if not DEFAULT_PATH.exists():
            st.warning("Dataset not found. Place the official UCI Online Retail file at `data/raw/Online Retail.xlsx`, or enable Upload a dataset in the sidebar.")
            st.info("Download: https://archive.ics.uci.edu/dataset/352/online+retail")
            st.stop()
        raw, prepared, audit = load_from_path(str(DEFAULT_PATH), DEFAULT_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

min_date = prepared["InvoiceDate"].min()
max_date = prepared["InvoiceDate"].max()
with st.sidebar:
    st.header("Filters")
    if pd.notna(min_date) and pd.notna(max_date):
        selected_dates = st.date_input("Invoice date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
    else:
        selected_dates = ()
    countries = sorted(prepared["Country"].dropna().astype(str).unique().tolist())
    selected_countries = st.multiselect("Countries", countries, default=[])

filtered = prepared.copy()
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start, end = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = filtered.loc[filtered["InvoiceDate"].between(start, end, inclusive="both")]
if selected_countries:
    filtered = filtered.loc[filtered["Country"].astype(str).isin(selected_countries)]

metrics = overview_metrics(filtered)
st.caption(f"Showing {len(filtered):,} of {len(prepared):,} input rows. Standard sales metrics include valid positive-quantity, positive-price lines whose invoice number does not start with C.")
cols = st.columns(4)
cols[0].metric("Input records", f"{metrics['input_rows']:,}")
cols[1].metric("Standard sale lines", f"{metrics['standard_sale_lines']:,}")
cols[2].metric("Recorded standard sales", f"{metrics['recorded_standard_sales_value']:,.2f}")
cols[3].metric("Identified customers", f"{metrics['identified_customers']:,}")
st.caption(f"Period: {metrics['start_date']} to {metrics['end_date']} · Customer ID coverage: {metrics['customer_id_coverage_pct']:.1f}% · {metrics['countries']} countries")
st.info("Recorded sales value is not profit. The dataset does not include product costs or operating expenses. Cancellation invoice prefixes are indicators, not proof of a specific return reason.")

tabs = st.tabs(["Overview", "Products & countries", "Customer segmentation", "Opportunity finder", "Data quality", "Customer behaviour"])
with tabs[0]:
    st.subheader("Recorded sales trend")
    frequency = st.radio("Trend granularity", ["Monthly", "Daily"], horizontal=True)
    trend = sales_over_time(filtered, "MS" if frequency == "Monthly" else "D")
    if trend.empty:
        st.warning("No standard sale lines match the selected filters.")
    else:
        fig = px.line(trend, x="Period", y="RecordedSalesValue", markers=True, title="Recorded standard sales value")
        fig.update_layout(xaxis_title="Period", yaxis_title="Recorded sales value")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View trend data"):
            st.dataframe(trend, use_container_width=True)
    cust = customer_concentration(filtered)
    if not cust.empty:
        st.subheader("Sales concentration among identified customers")
        cust = cust.sort_values("HistoricalSpend", ascending=False).reset_index(drop=True)
        cust["CumulativeSharePct"] = cust["HistoricalSpend"].cumsum() / cust["HistoricalSpend"].sum() * 100
        fig2 = px.line(cust.reset_index(), x=cust.reset_index().index + 1, y="CumulativeSharePct", title="Cumulative share of recorded standard sales by customer rank")
        fig2.update_layout(xaxis_title="Customer rank (highest historical spend first)", yaxis_title="Cumulative share (%)")
        st.plotly_chart(fig2, use_container_width=True)
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.subheader("Top products")
        products = group_performance(filtered, "Description", 15)
        if products.empty:
            st.warning("No product results for these filters.")
        else:
            fig = px.bar(products.sort_values("RecordedSalesValue"), x="RecordedSalesValue", y="Description", orientation="h", title="Top products by recorded sales")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(products, use_container_width=True)
    with right:
        st.subheader("Country performance")
        country = group_performance(filtered, "Country", 15)
        if country.empty:
            st.warning("No country results for these filters.")
        else:
            fig = px.bar(country.sort_values("RecordedSalesValue"), x="RecordedSalesValue", y="Country", orientation="h", title="Top countries by recorded sales")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(country, use_container_width=True)
with tabs[2]:
    st.subheader("Customer segmentation")
    st.write("Segments are assigned using explicit RFM rules so an analyst can explain why each customer belongs to a group. Percentile cutoffs are recalculated for the current filters.")
    rfm = build_rfm(filtered)
    if rfm.empty:
        st.warning("No identified customers with standard sale lines match these filters.")
    else:
        segmented = assign_customer_segments(rfm)
        summary = summarize_customer_segments(segmented)
        a, b, c = st.columns(3)
        a.metric("Identified customers", f"{len(segmented):,}")
        b.metric("Segments", f"{segmented['Segment'].nunique():,}")
        c.metric("Median historical spend", f"{segmented['MonetaryValue'].median():,.2f}")
        fig = px.bar(summary.sort_values("Customers"), x="Customers", y="Segment", orientation="h",
                     title="Customer segments by size", text="Customers")
        fig.update_layout(xaxis_title="Identified customers", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Segment profiles")
        st.dataframe(summary[["Segment", "Customers", "SharePct", "MedianRecencyDays", "MedianFrequency", "MedianHistoricalSpend", "TotalHistoricalSpend"]].round(2), use_container_width=True)
        st.subheader("What each segment means")
        for row in summary.itertuples(index=False):
            example = segmented.loc[segmented["Segment"] == row.Segment].iloc[0]
            with st.expander(f"{row.Segment} — {row.Customers:,} customers ({row.SharePct:.1f}%)"):
                st.write(example["SegmentReason"])
                st.write("**Suggested analyst action:** " + example["RecommendedAction"])
        st.subheader("Customer-level segment assignments")
        selected_segment = st.selectbox("Filter by segment", ["All segments"] + summary["Segment"].tolist())
        view = segmented if selected_segment == "All segments" else segmented.loc[segmented["Segment"] == selected_segment]
        st.dataframe(view[["CustomerID", "Segment", "SegmentReason", "RecommendedAction", "RecencyDays", "Frequency", "MonetaryValue", "DistinctProducts", "LastPurchase"]], use_container_width=True)
        st.download_button("Download customer segment assignments", view.to_csv(index=False).encode("utf-8"), "customer_segments.csv", "text/csv")
        st.caption("These are transparent descriptive segments based on the current analysis population, not predictions. Historical spend is not profit or lifetime value.")
with tabs[3]:
    st.subheader("Customer Opportunity Finder")
    rfm = build_rfm(filtered)
    opportunities = add_opportunity_flags(rfm)
    if opportunities.empty:
        st.warning("No identified customers available for opportunity analysis.")
    else:
        priority = opportunities.loc[opportunities["Opportunity"] != "No priority flag"].copy()
        st.caption("Rules use relative percentiles in the currently filtered population. Flags are leads for human review, not churn predictions or guaranteed revenue opportunities.")
        st.metric("Customers flagged for review", f"{len(priority):,}")
        if priority.empty:
            st.info("No customers met the current opportunity rules for this filtered population.")
        else:
            st.dataframe(priority.sort_values(["MonetaryValue", "RecencyDays"], ascending=False), use_container_width=True)
with tabs[4]:
    st.subheader("Data quality audit")
    st.write("The audit describes the current filtered dataset; the full-source audit can be generated with the CLI reporting command.")
    st.dataframe(data_quality_summary(filtered), use_container_width=True)
    missing = pd.DataFrame({"Column": list(audit["missing_values"].keys()), "Missing rows": list(audit["missing_values"].values())})
    st.subheader("Missing values in loaded source")
    st.dataframe(missing, use_container_width=True)
    summary = pd.DataFrame([{"Metric": k, "Value": str(v)} for k, v in audit.items() if k != "missing_values"])
    with st.expander("Full source audit details"):
        st.dataframe(summary, use_container_width=True)
    st.download_button("Download filtered data-quality summary", data=data_quality_summary(filtered).to_csv(index=False).encode("utf-8"), file_name="data_quality_summary.csv", mime="text/csv")

st.divider()
st.caption("Retail Customer Intelligence · Local-first analytics · Interpret findings in context; verify opportunities before taking action.")

with tabs[5]:
    st.subheader("Purchase behaviour")
    st.caption("These views describe observed patterns. They do not establish why a customer behaves a certain way.")

    invoices, intervals = purchase_interval_summary(filtered)
    if intervals.empty:
        st.info("Not enough repeat orders to calculate purchase intervals.")
    else:
        left, right = st.columns(2)
        left.metric("Customers with repeat orders", f"{len(intervals):,}")
        right.metric("Median gap between orders", f"{intervals['MedianDaysBetweenOrders'].median():.1f} days")
        st.dataframe(intervals.sort_values("MedianDaysBetweenOrders").head(20).round(1), use_container_width=True)
        st.download_button("Download purchase interval summary", intervals.to_csv(index=False).encode("utf-8"), "purchase_intervals.csv", "text/csv")

    st.subheader("Products bought together")
    pairs = top_product_pairs(filtered, limit=15)
    if pairs.empty:
        st.info("No invoices contain more than one distinct product in the selected data.")
    else:
        st.dataframe(pairs, use_container_width=True)
        st.caption("Pairs are counted when products appear in the same invoice. This does not prove that one product causes another to be purchased.")

    st.subheader("Do customer segments behave differently later?")
    horizon = st.select_slider("Future period to evaluate", options=[30, 60, 90, 120], value=60, key="segment_horizon")
    segment_results, validation = segment_future_outcomes(filtered, horizon_days=horizon)
    if validation.get("status") != "ok":
        st.info(validation.get("reason", "Not enough data for a chronological evaluation."))
    else:
        st.caption(f"Segments are built only from history through {validation['cutoff_date']}; outcomes are measured through {validation['holdout_end_date']}.")
        st.metric("Overall customer purchase rate in holdout", f"{validation['overall_purchase_rate_pct']:.1f}%")
        st.dataframe(segment_results.round(2), use_container_width=True)
        st.caption("This retrospective check helps compare segments against later outcomes. It is not a causal test or a trained prediction model.")
        customer_outcomes = validation["customer_outcomes"]
        st.download_button("Download segment holdout outcomes", customer_outcomes.to_csv(index=False).encode("utf-8"), "segment_holdout_outcomes.csv", "text/csv")

    st.subheader("Cancellation-invoice patterns")
    cancellation_months, cancellation_products = cancellation_summary(filtered)
    if cancellation_months.empty:
        st.info("No cancellation-invoice rows found in the selected data.")
    else:
        st.write("By month")
        st.dataframe(cancellation_months, use_container_width=True)
        st.write("Products with the most flagged lines")
        st.dataframe(cancellation_products, use_container_width=True)
        st.caption("Invoice numbers beginning with C are flagged for review; they are not guaranteed to represent a specific return reason.")

    st.divider()
    st.subheader("Monthly customer retention")
    cohort_sizes, retention = customer_cohort_retention(filtered)
    if cohort_sizes.empty or retention.empty:
        st.info("Not enough identified standard-sale activity to build cohorts.")
    else:
        st.dataframe(retention.style.format("{:.1f}"), use_container_width=True)
        st.caption("Each row is a first-observed-purchase-month cohort; values show the percentage active in each subsequent month. Later cohorts have fewer months of observation.")
        st.download_button("Download retention matrix", retention.to_csv().encode("utf-8"), "cohort_retention.csv", "text/csv")

    st.subheader("Simple repeat-purchase baseline")
    horizon = st.select_slider("Future observation window", options=[30, 60, 90, 120], value=90, key="repeat_horizon")
    result = temporal_repeat_purchase_validation(filtered, horizon_days=horizon)
    if result.get("status") != "ok":
        st.info(result.get("reason", "Not enough data for a chronological holdout."))
    else:
        a, b, c = st.columns(3)
        a.metric("Eligible prior customers", f"{result['eligible_prior_customers']:,}")
        b.metric("Purchased in holdout", f"{result['returned_customers']:,}")
        c.metric("Observed return rate", f"{result['overall_repeat_rate_pct']:.1f}%")
        comparison = pd.DataFrame([
            {"Prior behaviour": "2+ distinct invoices in history", "Customers": result["prior_frequency_2plus_customers"], "Holdout purchase rate (%)": result["prior_frequency_2plus_return_rate_pct"]},
            {"Prior behaviour": "1 distinct invoice in history", "Customers": result["prior_frequency_1_customers"], "Holdout purchase rate (%)": result["prior_frequency_1_return_rate_pct"]},
        ])
        st.dataframe(comparison.round(1), use_container_width=True)
        st.caption("A descriptive benchmark, not a trained model. This dataset covers about one year, so seasonality and the chosen cutoff can affect results.")
