"""Customer feature engineering and transparent opportunity flags."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_rfm(prepared: pd.DataFrame, analysis_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build RFM from standard sale lines; analysis date defaults to day after max date."""
    data = prepared.loc[
        prepared["is_standard_sale_line"] & prepared["CustomerID"].notna()
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=["CustomerID", "RecencyDays", "Frequency", "MonetaryValue", "DistinctProducts", "LastPurchase"])
    data["CustomerID"] = data["CustomerID"].astype("int64")
    if analysis_date is None:
        analysis_date = data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    else:
        analysis_date = pd.Timestamp(analysis_date).normalize()
    invoices = data.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        MonetaryValue=("StandardSalesValue", "sum"),
        DistinctProducts=("StockCode", "nunique"),
    ).reset_index()
    invoices["RecencyDays"] = (analysis_date - invoices["LastPurchase"].dt.normalize()).dt.days
    invoices["Frequency"] = invoices["Frequency"].astype(int)
    invoices["DistinctProducts"] = invoices["DistinctProducts"].fillna(0).astype(int)
    return invoices[["CustomerID", "RecencyDays", "Frequency", "MonetaryValue", "DistinctProducts", "LastPurchase"]].sort_values("MonetaryValue", ascending=False)



def assign_customer_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign transparent descriptive segments using RFM percentile rules."""
    out = rfm.copy()
    if out.empty:
        out["Segment"] = pd.Series(dtype="string")
        out["SegmentReason"] = pd.Series(dtype="string")
        out["RecommendedAction"] = pd.Series(dtype="string")
        return out

    spend_hi = out["MonetaryValue"].quantile(0.75)
    spend_lo = out["MonetaryValue"].quantile(0.25)
    recent_cut = out["RecencyDays"].quantile(0.25)
    stale_cut = out["RecencyDays"].quantile(0.75)
    freq_hi = out["Frequency"].quantile(0.75)
    segments, reasons, actions = [], [], []

    for row in out.itertuples(index=False):
        if row.MonetaryValue >= spend_hi and row.RecencyDays >= stale_cut:
            seg = "High Value — Needs Review"
            reason = f"Spend is at/above the 75th percentile ({spend_hi:.2f}); recency is at/above {stale_cut:.0f} days."
            action = "Review account history and seasonality before considering reactivation outreach."
        elif row.MonetaryValue >= spend_hi and row.RecencyDays <= recent_cut:
            seg = "High Value — Recently Active"
            reason = "High historical spend and among the most recently active customers in this population."
            action = "Protect the relationship; explore relevant replenishment or complementary products."
        elif row.Frequency >= freq_hi and row.Frequency > 1:
            seg = "Repeat Customers"
            reason = f"Invoice frequency is at/above the 75th percentile ({freq_hi:.0f})."
            action = "Study repeat-purchase intervals and product affinity; do not assume future purchases."
        elif row.RecencyDays >= stale_cut:
            seg = "Low Recent Activity"
            reason = f"Time since last observed purchase is at/above {stale_cut:.0f} days."
            action = "Check whether the gap is unusual for this customer before taking action."
        elif row.MonetaryValue <= spend_lo and row.Frequency <= out["Frequency"].median():
            seg = "Developing Customers"
            reason = "Lower historical spend and no above-median invoice frequency in this population."
            action = "Study first-to-second purchase behavior and product preferences."
        else:
            seg = "Steady / Other Customers"
            reason = "Does not meet the current high-value, high-frequency, or long-gap rules."
            action = "Monitor aggregate behavior; do not target solely from this label."
        segments.append(seg)
        reasons.append(reason)
        actions.append(action)

    out["Segment"] = segments
    out["SegmentReason"] = reasons
    out["RecommendedAction"] = actions
    out["SegmentationNote"] = (
        "Rule-based descriptive segment using quantiles in the current analysis population; "
        "not a churn prediction or evidence of causal impact."
    )
    return out


def summarize_customer_segments(segmented: pd.DataFrame) -> pd.DataFrame:
    """Summarize segment size and behavior."""
    columns = ["Segment", "Customers", "SharePct", "MedianRecencyDays",
               "MedianFrequency", "MedianHistoricalSpend", "TotalHistoricalSpend"]
    if segmented.empty or "Segment" not in segmented.columns:
        return pd.DataFrame(columns=columns)
    total = segmented["CustomerID"].nunique()
    summary = segmented.groupby("Segment", as_index=False).agg(
        Customers=("CustomerID", "nunique"),
        MedianRecencyDays=("RecencyDays", "median"),
        MedianFrequency=("Frequency", "median"),
        MedianHistoricalSpend=("MonetaryValue", "median"),
        TotalHistoricalSpend=("MonetaryValue", "sum"),
    )
    summary["SharePct"] = summary["Customers"] / max(total, 1) * 100
    return summary[columns].sort_values("Customers", ascending=False)

def add_opportunity_flags(rfm: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent relative rules; flags are leads for review, not churn predictions."""
    out = rfm.copy()
    if out.empty:
        out["Opportunity"] = pd.Series(dtype="string")
        out["Reason"] = pd.Series(dtype="string")
        out["SuggestedAction"] = pd.Series(dtype="string")
        return out
    spend_cut = out["MonetaryValue"].quantile(0.75)
    recency_cut = out["RecencyDays"].quantile(0.75)
    frequency_cut = out["Frequency"].quantile(0.75)
    labels, reasons, actions = [], [], []
    for row in out.itertuples(index=False):
        if row.MonetaryValue >= spend_cut and row.RecencyDays >= recency_cut:
            labels.append("High historical spend; relatively long gap")
            reasons.append(f"Historical spend is at or above the 75th percentile; recency is at or above {recency_cut:.0f} days (75th percentile).")
            actions.append("Review account history and validate whether the gap is unusual before outreach.")
        elif row.Frequency >= frequency_cut and row.Frequency > 1:
            labels.append("Frequent historical purchaser")
            reasons.append(f"Observed {row.Frequency} distinct invoices; frequency is at or above the dataset's 75th percentile ({frequency_cut:.0f}).")
            actions.append("Review product preferences and consider a relevant repeat-purchase or cross-sell analysis.")
        else:
            labels.append("No priority flag")
            reasons.append("This customer does not meet the current transparent opportunity rules.")
            actions.append("No targeted action suggested by this rule set.")
    out["Opportunity"] = labels
    out["Reason"] = reasons
    out["SuggestedAction"] = actions
    out["RuleNote"] = "Relative historical-data flag; not a churn prediction or evidence of causal impact."
    return out


def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Fit exploratory K-means on log-scaled monetary/frequency features.

    Recency is retained as a descriptive output but excluded from clustering to
    keep this minimal version interpretable. Clusters are exploratory, not labels
    of customer intent. Returns unchanged input with Cluster column and diagnostics.
    """
    out = rfm.copy()
    features = ["RecencyDays", "Frequency", "MonetaryValue", "DistinctProducts"]
    if len(out) < max(3, n_clusters) or any(c not in out.columns for c in features):
        out["Cluster"] = -1
        return out, {"status": "insufficient_data", "rows": int(len(out))}
    X = out[features].astype(float).copy()
    for col in ["Frequency", "MonetaryValue", "DistinctProducts"]:
        X[col] = np.log1p(X[col].clip(lower=0))
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    if valid.sum() <= n_clusters or X.loc[valid].nunique().min() <= 1:
        out["Cluster"] = -1
        return out, {"status": "insufficient_variation", "rows": int(valid.sum())}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.loc[valid])
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    out["Cluster"] = -1
    out.loc[valid, "Cluster"] = labels
    unique_labels = np.unique(labels)
    diagnostics: dict[str, float | int | str] = {"status": "ok", "rows": int(valid.sum()), "clusters": int(len(unique_labels))}
    if len(unique_labels) > 1 and len(unique_labels) < len(labels):
        # Silhouette is expensive on the full set; deterministic subsample for large sets.
        if len(labels) > 5000:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(labels), 5000, replace=False)
            score = silhouette_score(X_scaled[idx], labels[idx])
            diagnostics["silhouette_sample_size"] = 5000
        else:
            score = silhouette_score(X_scaled, labels)
        diagnostics["silhouette_score"] = float(score)
    return out, diagnostics
