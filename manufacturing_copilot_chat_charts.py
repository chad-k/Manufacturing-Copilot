import re
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


st.set_page_config(
    page_title="Manufacturing Copilot Prototype",
    page_icon="🏭",
    layout="wide",
)


_PLOTLY_COUNTER = 0

def show_chart(fig, prefix="chart"):
    global _PLOTLY_COUNTER
    _PLOTLY_COUNTER += 1
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_{_PLOTLY_COUNTER}")


REQUIRED_COLUMNS = {
    "Timestamp",
    "Part",
    "Machine",
    "Characteristic",
    "Value",
    "LSL",
    "Target",
    "USL",
}


@st.cache_data

def generate_demo_data(rows_per_combo: int = 180) -> pd.DataFrame:
    """Generate deterministic manufacturing data for a working demo."""
    rng = np.random.default_rng(42)
    end_time = pd.Timestamp.now().floor("h")

    specs = {
        "Length": (9.50, 10.00, 10.50),
        "Weight": (48.00, 50.00, 52.00),
        "Thickness": (1.80, 2.00, 2.20),
    }

    rows = []
    parts = ["P1", "P2"]
    machines = ["M1", "M2"]

    for part in parts:
        for machine in machines:
            for characteristic, (lsl, target, usl) in specs.items():
                timestamps = pd.date_range(
                    end=end_time,
                    periods=rows_per_combo,
                    freq="4h",
                )

                part_shift = 0.0 if part == "P1" else (usl - lsl) * 0.035
                machine_shift = 0.0 if machine == "M1" else (usl - lsl) * 0.055
                sigma = (usl - lsl) / (7.2 if machine == "M1" else 5.8)

                values = rng.normal(
                    target + part_shift + machine_shift,
                    sigma,
                    rows_per_combo,
                )

                # Add visible manufacturing behavior for the demo.
                if part == "P1" and machine == "M2" and characteristic == "Length":
                    values[-35:] += np.linspace(0, (usl - lsl) * 0.16, 35)
                if part == "P2" and machine == "M2" and characteristic == "Weight":
                    values[-8:] += (usl - lsl) * 0.15
                if part == "P2" and machine == "M1" and characteristic == "Thickness":
                    values[-25:] -= (usl - lsl) * 0.08

                for ts, value in zip(timestamps, values):
                    rows.append(
                        {
                            "Timestamp": ts,
                            "Part": part,
                            "Machine": machine,
                            "Characteristic": characteristic,
                            "Value": round(float(value), 4),
                            "LSL": lsl,
                            "Target": target,
                            "USL": usl,
                        }
                    )

    df = pd.DataFrame(rows)
    df["Status"] = np.where(
        (df["Value"] < df["LSL"]) | (df["Value"] > df["USL"]),
        "Out of Spec",
        "In Spec",
    )
    return df.sort_values("Timestamp").reset_index(drop=True)


@st.cache_data

def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    numeric_columns = ["Value", "LSL", "Target", "USL"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Timestamp", *numeric_columns])
    df["Part"] = df["Part"].astype(str)
    df["Machine"] = df["Machine"].astype(str)
    df["Characteristic"] = df["Characteristic"].astype(str)
    df["Status"] = np.where(
        (df["Value"] < df["LSL"]) | (df["Value"] > df["USL"]),
        "Out of Spec",
        "In Spec",
    )
    return df.sort_values("Timestamp").reset_index(drop=True)


def capability(group: pd.DataFrame) -> pd.Series:
    values = group["Value"].dropna()
    mean = values.mean()
    std = values.std(ddof=1)
    lsl = group["LSL"].iloc[0]
    target = group["Target"].iloc[0]
    usl = group["USL"].iloc[0]

    if len(values) < 2 or pd.isna(std) or std <= 0:
        cp = np.nan
        cpk = np.nan
    else:
        cp = (usl - lsl) / (6 * std)
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)

    return pd.Series(
        {
            "Count": len(values),
            "Mean": mean,
            "StdDev": std,
            "Cp": cp,
            "Cpk": cpk,
            "LSL": lsl,
            "Target": target,
            "USL": usl,
            "OOS Count": int(((values < lsl) | (values > usl)).sum()),
            "OOS Rate %": float(((values < lsl) | (values > usl)).mean() * 100),
        }
    )


def calculate_capability_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    table = (
        df.groupby(["Part", "Machine", "Characteristic"], dropna=False)
        .apply(capability)
        .reset_index()
    )
    return table.sort_values("Cpk", na_position="last").reset_index(drop=True)


def detect_nelson_rules(values: pd.Series) -> pd.DataFrame:
    """Detect a useful subset of Nelson rules for an individual-value demo."""
    series = values.reset_index(drop=True).astype(float)
    mean = series.mean()
    std = series.std(ddof=1)
    records = []

    if len(series) < 2 or pd.isna(std) or std <= 0:
        return pd.DataFrame(columns=["Point", "Rule", "Description"])

    # Rule 1: one point beyond 3 sigma.
    for i, value in enumerate(series):
        if abs(value - mean) > 3 * std:
            records.append((i, "Rule 1", "One point beyond 3 standard deviations"))

    # Rule 2: nine consecutive points on the same side of the mean.
    for start in range(0, len(series) - 8):
        window = series.iloc[start : start + 9]
        if (window > mean).all() or (window < mean).all():
            records.append((start + 8, "Rule 2", "Nine points on one side of the mean"))

    # Rule 3: six consecutive increasing or decreasing points.
    for start in range(0, len(series) - 5):
        window = series.iloc[start : start + 6]
        diffs = np.diff(window)
        if (diffs > 0).all() or (diffs < 0).all():
            records.append((start + 5, "Rule 3", "Six points steadily increasing or decreasing"))

    result = pd.DataFrame(records, columns=["Point", "Rule", "Description"])
    return result.drop_duplicates().sort_values("Point") if not result.empty else result


def extract_entity(question: str, choices) -> Optional[str]:
    upper = question.upper()
    for choice in sorted(map(str, choices), key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(choice.upper())}(?![A-Z0-9])", upper):
            return choice
    return None


def extract_limit(question: str, default: int = 5) -> int:
    match = re.search(r"\b(?:top|worst|best)\s+(\d+)\b", question.lower())
    if match:
        return max(1, min(int(match.group(1)), 20))
    return default


def parse_question(question: str, df: pd.DataFrame) -> Dict[str, Optional[str]]:
    q = question.lower().strip()

    if any(word in q for word in ["help", "what can", "examples", "commands"]):
        intent = "help"
    elif "compare" in q or "versus" in q or " vs " in f" {q} ":
        intent = "compare"
    elif "nelson" in q or "rule violation" in q or "out of control" in q:
        intent = "nelson"
    elif "trend" in q or "over time" in q or "control chart" in q:
        intent = "trend"
    elif "worst" in q and "cpk" in q:
        intent = "worst_cpk"
    elif "best" in q and "cpk" in q:
        intent = "best_cpk"
    elif "cpk" in q or "capability" in q:
        intent = "capability"
    elif "out of spec" in q or "oos" in q or "defect" in q or "fail" in q:
        intent = "oos"
    else:
        intent = "summary"

    return {
        "intent": intent,
        "part": extract_entity(question, df["Part"].unique()),
        "machine": extract_entity(question, df["Machine"].unique()),
        "characteristic": extract_entity(question, df["Characteristic"].unique()),
        "limit": extract_limit(question),
    }


def filter_data(
    df: pd.DataFrame,
    part: Optional[str],
    machine: Optional[str],
    characteristic: Optional[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    mask = (
        (df["Timestamp"].dt.date >= start_date)
        & (df["Timestamp"].dt.date <= end_date)
    )
    if part:
        mask &= df["Part"] == part
    if machine:
        mask &= df["Machine"] == machine
    if characteristic:
        mask &= df["Characteristic"] == characteristic
    return df.loc[mask].copy()


def add_capability_reference_lines(fig: go.Figure) -> None:
    fig.add_hline(y=1.00, line_dash="dot", annotation_text="Cpk 1.00")
    fig.add_hline(y=1.33, line_dash="dash", annotation_text="Cpk 1.33")


def render_capability(df: pd.DataFrame, title: str = "Capability Results") -> None:
    table = calculate_capability_table(df)
    if table.empty:
        st.warning("No data matched that request.")
        return

    lowest = table.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest Cpk", "N/A" if pd.isna(lowest["Cpk"]) else f"{lowest['Cpk']:.2f}")
    c2.metric("Groups Reviewed", len(table))
    c3.metric("Below Cpk 1.33", int((table["Cpk"] < 1.33).sum()))
    c4.metric("Out-of-Spec Points", int(table["OOS Count"].sum()))

    plot_df = table.dropna(subset=["Cpk"]).copy()
    plot_df["Process"] = (
        plot_df["Part"] + " / " + plot_df["Machine"] + " / " + plot_df["Characteristic"]
    )
    fig = px.bar(plot_df, x="Process", y="Cpk", title=title)
    add_capability_reference_lines(fig)
    show_chart(fig)
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)


def render_trend(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data matched that request.")
        return

    combos = df[["Part", "Machine", "Characteristic"]].drop_duplicates()
    if len(combos) > 1:
        st.info("Multiple processes matched. Showing the first one; specify part, machine, and characteristic for a focused chart.")
        first = combos.iloc[0]
        df = df[
            (df["Part"] == first["Part"])
            & (df["Machine"] == first["Machine"])
            & (df["Characteristic"] == first["Characteristic"])
        ]

    df = df.sort_values("Timestamp").copy()
    mean = df["Value"].mean()
    std = df["Value"].std(ddof=1)
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Timestamp"], y=df["Value"], mode="lines+markers", name="Value"))
    fig.add_hline(y=df["Target"].iloc[0], line_dash="dash", annotation_text="Target")
    fig.add_hline(y=df["USL"].iloc[0], line_dash="dot", annotation_text="USL")
    fig.add_hline(y=df["LSL"].iloc[0], line_dash="dot", annotation_text="LSL")
    fig.add_hline(y=ucl, line_dash="dashdot", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dashdot", annotation_text="LCL")
    fig.update_layout(
        title=f"Trend: {df['Part'].iloc[0]} / {df['Machine'].iloc[0]} / {df['Characteristic'].iloc[0]}",
        xaxis_title="Time",
        yaxis_title="Measurement",
    )
    show_chart(fig)

    recent = df.tail(20)
    delta = recent["Value"].iloc[-1] - recent["Value"].iloc[0] if len(recent) > 1 else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Value", f"{df['Value'].iloc[-1]:.4f}")
    c2.metric("Recent Change", f"{delta:+.4f}")
    c3.metric("Recent OOS", int((recent["Status"] == "Out of Spec").sum()))


def render_nelson(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data matched that request.")
        return

    combos = df[["Part", "Machine", "Characteristic"]].drop_duplicates()
    summary_rows = []
    violation_rows = []

    for _, combo in combos.iterrows():
        group = df[
            (df["Part"] == combo["Part"])
            & (df["Machine"] == combo["Machine"])
            & (df["Characteristic"] == combo["Characteristic"])
        ].sort_values("Timestamp")
        rules = detect_nelson_rules(group["Value"])
        summary_rows.append(
            {
                "Part": combo["Part"],
                "Machine": combo["Machine"],
                "Characteristic": combo["Characteristic"],
                "Violations": len(rules),
            }
        )
        if not rules.empty:
            rules = rules.copy()
            rules["Timestamp"] = rules["Point"].map(group.reset_index(drop=True)["Timestamp"])
            rules["Value"] = rules["Point"].map(group.reset_index(drop=True)["Value"])
            rules["Part"] = combo["Part"]
            rules["Machine"] = combo["Machine"]
            rules["Characteristic"] = combo["Characteristic"]
            violation_rows.append(rules)

    summary = pd.DataFrame(summary_rows).sort_values("Violations", ascending=False)
    st.metric("Total Nelson Violations", int(summary["Violations"].sum()))
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if violation_rows:
        details = pd.concat(violation_rows, ignore_index=True)
        st.subheader("Violation Details")
        st.dataframe(
            details[["Timestamp", "Part", "Machine", "Characteristic", "Rule", "Description", "Value"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No Nelson Rule 1, 2, or 3 violations were detected in the selected data.")


def render_compare(df: pd.DataFrame) -> None:
    table = calculate_capability_table(df)
    if table.empty:
        st.warning("No data matched that request.")
        return

    group_col = "Machine" if table["Machine"].nunique() > 1 else "Part"
    fig = px.box(
        df,
        x=group_col,
        y="Value",
        points="outliers",
        facet_col="Characteristic" if df["Characteristic"].nunique() > 1 else None,
        title=f"Process Comparison by {group_col}",
    )
    show_chart(fig)
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)


def render_oos(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data matched that request.")
        return

    grouped = (
        df.assign(OOS=(df["Status"] == "Out of Spec").astype(int))
        .groupby(["Part", "Machine", "Characteristic"], as_index=False)
        .agg(Measurements=("Value", "size"), OOS_Count=("OOS", "sum"))
    )
    grouped["OOS_Rate_%"] = grouped["OOS_Count"] / grouped["Measurements"] * 100
    grouped = grouped.sort_values("OOS_Rate_%", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Measurements", len(df))
    c2.metric("Out-of-Spec", int((df["Status"] == "Out of Spec").sum()))
    c3.metric("OOS Rate", f"{(df['Status'] == 'Out of Spec').mean() * 100:.2f}%")

    fig = px.bar(
        grouped,
        x="Characteristic",
        y="OOS_Rate_%",
        color="Machine",
        barmode="group",
        hover_data=["Part", "OOS_Count", "Measurements"],
        title="Out-of-Spec Rate",
    )
    show_chart(fig)
    st.dataframe(grouped.round(3), use_container_width=True, hide_index=True)


def render_summary(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data matched that request.")
        return
    st.write("Here is a manufacturing quality summary for the selected period.")
    render_capability(df, "Capability Overview")




def extract_entities(question: str, choices) -> list:
    upper = question.upper()
    found = []
    for choice in sorted(map(str, choices), key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(choice.upper())}(?![A-Z0-9])", upper):
            found.append(choice)
    return found


def requested_chart(question: str) -> Optional[str]:
    q = question.lower()
    aliases = [
        ("multiple_data_table", ["multiple data table", "multiple tables"]),
        ("combined_control", ["combined control chart"]),
        ("normal_probability", ["normal probability", "probability plot", "q-q plot", "qq plot"]),
        ("individual_mr", ["individuals and moving range", "individual and moving range", "i-mr", "imr"]),
        ("xbar_r", ["x-bar and r", "xbar r", "x-bar r"]),
        ("xbar_s", ["x-bar and s", "xbar s", "x-bar s"]),
        ("moving_average", ["moving average"]),
        ("time_axis", ["time axis"]),
        ("control", ["control chart"]),
        ("histogram", ["histogram", "distribution chart"]),
        ("statistics", ["statistics list", "stats table", "statistics", "descriptive stats"]),
        ("data_table", ["data table", "raw data", "show rows", "show records"]),
        ("combination", ["combination chart", "combo chart"]),
        ("scatter", ["scatter chart", "scatter plot", "correlation plot"]),
        ("trend", ["trend chart", "trend", "over time"]),
        ("ewma", ["ewma"]),
        ("cusum", ["cusum"]),
        ("performance", ["performance chart"]),
        ("capability", ["process capability", "capability chart", "cpk chart"]),
        ("sparkline", ["sparkline"]),
        ("monitor", ["monitor table"]),
    ]
    for name, terms in aliases:
        if any(term in q for term in terms):
            return name
    return None


def parse_question_v2(question: str, df: pd.DataFrame) -> Dict:
    q = question.lower().strip()
    chart = requested_chart(question)
    parts = extract_entities(question, df["Part"].unique())
    machines = extract_entities(question, df["Machine"].unique())
    traces = extract_entities(question, df["Characteristic"].unique())

    if any(x in q for x in ["help", "what can", "examples", "commands"]):
        intent = "help"
    elif any(x in q for x in ["dashboard", "scorecard", "overview page"]):
        intent = "dashboard"
    elif "compare" in q or "versus" in q or re.search(r"\bvs\.?\b", q):
        intent = "compare"
    elif chart:
        intent = "chart"
    elif "nelson" in q or "rule violation" in q or "out of control" in q:
        intent = "nelson"
    elif "worst" in q and "cpk" in q:
        intent = "worst_cpk"
    elif "best" in q and "cpk" in q:
        intent = "best_cpk"
    elif "cpk" in q or "capability" in q:
        intent = "capability"
    elif "out of spec" in q or "oos" in q or "fail" in q:
        intent = "oos"
    else:
        intent = "summary"

    compare_dimension = None
    if "machine" in q:
        compare_dimension = "Machine"
    elif "part" in q:
        compare_dimension = "Part"
    elif any(x in q for x in ["trace", "characteristic", "measurement"]):
        compare_dimension = "Characteristic"

    return {
        "intent": intent,
        "chart": chart,
        "parts": parts,
        "machines": machines,
        "traces": traces,
        "limit": extract_limit(question),
        "compare_dimension": compare_dimension,
    }


def filter_data_v2(df, parts, machines, traces, start_date, end_date):
    mask = ((df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date))
    if parts:
        mask &= df["Part"].isin(parts)
    if machines:
        mask &= df["Machine"].isin(machines)
    if traces:
        mask &= df["Characteristic"].isin(traces)
    return df.loc[mask].copy()


def process_name(df):
    return df["Part"] + " / " + df["Machine"] + " / " + df["Characteristic"]


def one_process(df):
    if df.empty:
        return df
    combos = df[["Part", "Machine", "Characteristic"]].drop_duplicates()
    if len(combos) > 1:
        st.info("This chart requires one process. Showing the first matching part, machine, and trace.")
    first = combos.iloc[0]
    return df[(df["Part"] == first["Part"]) & (df["Machine"] == first["Machine"]) & (df["Characteristic"] == first["Characteristic"])].copy()


def render_control(df):
    d = one_process(df).sort_values("Timestamp")
    if d.empty:
        st.warning("No data matched that request."); return
    mean = d["Value"].mean(); std = d["Value"].std(ddof=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["Value"], mode="lines+markers", name="Value"))
    for y, label, dash in [(mean, "Center", "dash"), (mean+3*std, "UCL", "dot"), (mean-3*std, "LCL", "dot"), (d["Target"].iloc[0], "Target", "dash"), (d["USL"].iloc[0], "USL", "dot"), (d["LSL"].iloc[0], "LSL", "dot")]:
        fig.add_hline(y=y, line_dash=dash, annotation_text=label)
    fig.update_layout(title="Control Chart", xaxis_title="Time", yaxis_title="Measurement")
    show_chart(fig, "control")


def render_histogram(df):
    if df.empty: st.warning("No data matched that request."); return
    d=df.copy(); d["Process"]=process_name(d)
    fig=px.histogram(d, x="Value", color="Process" if d["Process"].nunique()>1 else None, marginal="box", nbins=35, opacity=.7, title="Histogram")
    show_chart(fig, "hist")


def render_statistics(df):
    table=calculate_capability_table(df)
    if table.empty: st.warning("No data matched that request."); return
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)


def render_data_table(df):
    if df.empty: st.warning("No data matched that request."); return
    st.caption(f"{len(df):,} measurement rows")
    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True)


def subgroup_data(df, size=5):
    d=one_process(df).sort_values("Timestamp").reset_index(drop=True)
    if d.empty: return d
    d["Subgroup"]=np.arange(len(d))//size
    g=d.groupby("Subgroup", as_index=False).agg(Timestamp=("Timestamp","max"), Xbar=("Value","mean"), Range=("Value",lambda s:s.max()-s.min()), StdDev=("Value","std"), Count=("Value","size"))
    return g[g["Count"]>=2]


def render_xbar_r(df):
    g=subgroup_data(df)
    if g.empty: st.warning("Not enough data for X-Bar and R."); return
    xb=g["Xbar"].mean(); rb=g["Range"].mean(); A2=.577; D3=0; D4=2.114
    fig=make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("X-Bar", "Range"))
    fig.add_trace(go.Scatter(x=g["Timestamp"],y=g["Xbar"],mode="lines+markers",name="X-Bar"),row=1,col=1)
    for y in [xb, xb+A2*rb, xb-A2*rb]: fig.add_hline(y=y,row=1,col=1)
    fig.add_trace(go.Scatter(x=g["Timestamp"],y=g["Range"],mode="lines+markers",name="Range"),row=2,col=1)
    for y in [rb, D4*rb, D3*rb]: fig.add_hline(y=y,row=2,col=1)
    fig.update_layout(height=700,title="X-Bar and R Chart"); show_chart(fig,"xbar_r")


def render_xbar_s(df):
    g=subgroup_data(df)
    if g.empty: st.warning("Not enough data for X-Bar and S."); return
    xb=g["Xbar"].mean(); sb=g["StdDev"].mean(); A3=1.427; B3=0; B4=2.089
    fig=make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("X-Bar", "S"))
    fig.add_trace(go.Scatter(x=g["Timestamp"],y=g["Xbar"],mode="lines+markers",name="X-Bar"),row=1,col=1)
    for y in [xb, xb+A3*sb, xb-A3*sb]: fig.add_hline(y=y,row=1,col=1)
    fig.add_trace(go.Scatter(x=g["Timestamp"],y=g["StdDev"],mode="lines+markers",name="S"),row=2,col=1)
    for y in [sb, B4*sb, B3*sb]: fig.add_hline(y=y,row=2,col=1)
    fig.update_layout(height=700,title="X-Bar and S Chart"); show_chart(fig,"xbar_s")


def render_imr(df):
    d=one_process(df).sort_values("Timestamp").reset_index(drop=True)
    if d.empty: st.warning("No data matched that request."); return
    d["MR"]=d["Value"].diff().abs(); mr=d["MR"].mean(); mean=d["Value"].mean(); sigma=mr/1.128 if mr>0 else d["Value"].std(ddof=1)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Individuals","Moving Range"))
    fig.add_trace(go.Scatter(x=d["Timestamp"],y=d["Value"],mode="lines+markers",name="Value"),row=1,col=1)
    for y in [mean,mean+3*sigma,mean-3*sigma]: fig.add_hline(y=y,row=1,col=1)
    fig.add_trace(go.Scatter(x=d["Timestamp"],y=d["MR"],mode="lines+markers",name="MR"),row=2,col=1)
    for y in [mr,3.267*mr]: fig.add_hline(y=y,row=2,col=1)
    fig.update_layout(height=700,title="Individuals and Moving Range"); show_chart(fig,"imr")


def render_scatter(df):
    if df.empty: st.warning("No data matched that request."); return
    traces=list(df["Characteristic"].unique())
    if len(traces)>=2:
        p=df.pivot_table(index=["Timestamp","Part","Machine"],columns="Characteristic",values="Value",aggfunc="mean").dropna().reset_index()
        if not p.empty:
            fig=px.scatter(p,x=traces[0],y=traces[1],color="Machine",symbol="Part",title=f"{traces[0]} vs {traces[1]}")
            show_chart(fig,"scatter"); return
    d=df.copy(); d["Sequence"]=d.groupby(["Part","Machine","Characteristic"]).cumcount()
    fig=px.scatter(d,x="Sequence",y="Value",color="Machine",symbol="Part",facet_col="Characteristic" if d["Characteristic"].nunique()>1 else None,title="Scatter Chart")
    show_chart(fig,"scatter")


def render_trend_multi(df):
    if df.empty: st.warning("No data matched that request."); return
    d=df.sort_values("Timestamp").copy(); d["Process"]=process_name(d)
    fig=px.line(d,x="Timestamp",y="Value",color="Process",markers=True,title="Trend Chart")
    show_chart(fig,"trend")


def render_ewma(df):
    d=one_process(df).sort_values("Timestamp").copy()
    if d.empty: st.warning("No data matched that request."); return
    d["EWMA"]=d["Value"].ewm(alpha=.2,adjust=False).mean(); mean=d["Value"].mean(); std=d["Value"].std(ddof=1)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=d["Timestamp"],y=d["Value"],mode="markers",name="Value",opacity=.35)); fig.add_trace(go.Scatter(x=d["Timestamp"],y=d["EWMA"],mode="lines",name="EWMA")); fig.add_hline(y=mean+3*std,line_dash="dot"); fig.add_hline(y=mean-3*std,line_dash="dot"); fig.update_layout(title="EWMA Chart"); show_chart(fig,"ewma")


def render_moving_average(df):
    if df.empty: st.warning("No data matched that request."); return
    d=df.sort_values("Timestamp").copy(); d["Process"]=process_name(d); d["Moving Average"]=d.groupby("Process")["Value"].transform(lambda s:s.rolling(10,min_periods=1).mean())
    fig=px.line(d,x="Timestamp",y="Moving Average",color="Process",title="10-Point Moving Average"); show_chart(fig,"ma")


def render_time_axis(df):
    if df.empty: st.warning("No data matched that request."); return
    d=df.copy(); d["Date"]=d["Timestamp"].dt.date; d["Hour"]=d["Timestamp"].dt.hour.astype(str)
    g=d.groupby(["Date","Hour"],as_index=False).agg(Mean=("Value","mean"))
    fig=px.line(g,x="Date",y="Mean",color="Hour",markers=True,title="Time Axis Chart"); show_chart(fig,"time")


def render_cusum(df):
    d=one_process(df).sort_values("Timestamp").copy()
    if d.empty: st.warning("No data matched that request."); return
    target=d["Target"].iloc[0]; std=d["Value"].std(ddof=1); k=.5*std; cp=[]; cm=[]; p=m=0
    for v in d["Value"]:
        p=max(0,p+v-target-k); m=min(0,m+v-target+k); cp.append(p); cm.append(m)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=d["Timestamp"],y=cp,name="CUSUM+")); fig.add_trace(go.Scatter(x=d["Timestamp"],y=cm,name="CUSUM-")); fig.add_hline(y=0); fig.update_layout(title="CUSUM Chart"); show_chart(fig,"cusum")


def render_normal_probability(df):
    d=one_process(df)
    if d.empty: st.warning("No data matched that request."); return
    vals=np.sort(d["Value"].to_numpy()); n=len(vals); probs=(np.arange(1,n+1)-.5)/n
    theoretical=scipy_stats.norm.ppf(probs) if scipy_stats is not None else np.linspace(-2.8,2.8,n)
    fit=np.polyfit(theoretical,vals,1); line=fit[0]*theoretical+fit[1]
    fig=go.Figure(); fig.add_trace(go.Scatter(x=theoretical,y=vals,mode="markers",name="Observed")); fig.add_trace(go.Scatter(x=theoretical,y=line,mode="lines",name="Reference")); fig.update_layout(title="Normal Probability Plot",xaxis_title="Theoretical Quantile",yaxis_title="Observed Value"); show_chart(fig,"prob")


def render_performance(df):
    t=calculate_capability_table(df)
    if t.empty: st.warning("No data matched that request."); return
    t["Process"]=t["Part"]+" / "+t["Machine"]+" / "+t["Characteristic"]
    fig=px.bar(t,x="Process",y=["Cp","Cpk"],barmode="group",title="Performance Chart"); fig.add_hline(y=1.33,line_dash="dash"); show_chart(fig,"perf")


def render_sparkline(df):
    if df.empty: st.warning("No data matched that request."); return
    d=df.sort_values("Timestamp").copy(); d["Process"]=process_name(d); names=list(d["Process"].unique())[:8]
    fig=make_subplots(rows=len(names),cols=1,shared_xaxes=True,subplot_titles=names,vertical_spacing=.04)
    for i,name in enumerate(names,1):
        g=d[d["Process"]==name]; fig.add_trace(go.Scatter(x=g["Timestamp"],y=g["Value"],mode="lines",showlegend=False),row=i,col=1)
    fig.update_layout(height=max(300,130*len(names)),title="Sparkline Charts"); show_chart(fig,"spark")


def render_combined_control(df):
    if df.empty: st.warning("No data matched that request."); return
    d=df.sort_values("Timestamp").copy(); d["Process"]=process_name(d); d["Z"]=d.groupby("Process")["Value"].transform(lambda s:(s-s.mean())/s.std(ddof=1) if s.std(ddof=1)>0 else 0)
    fig=px.line(d,x="Timestamp",y="Z",color="Process",title="Combined Standardized Control Chart"); fig.add_hline(y=0); fig.add_hline(y=3,line_dash="dot"); fig.add_hline(y=-3,line_dash="dot"); show_chart(fig,"combined")


def render_monitor(df):
    t=calculate_capability_table(df)
    if t.empty: st.warning("No data matched that request."); return
    latest=df.sort_values("Timestamp").groupby(["Part","Machine","Characteristic"],as_index=False).tail(1)[["Part","Machine","Characteristic","Timestamp","Value","Status"]]
    m=t.merge(latest,on=["Part","Machine","Characteristic"],how="left"); m["Health"]=np.select([m["Status"].eq("Out of Spec"),m["Cpk"].lt(1),m["Cpk"].lt(1.33)],["Critical","Poor","Watch"],default="Good")
    st.dataframe(m[["Health","Part","Machine","Characteristic","Timestamp","Value","Status","Cpk","OOS Rate %"]].round(4),use_container_width=True,hide_index=True)


def render_multiple_tables(df):
    if df.empty: st.warning("No data matched that request."); return
    for (p,m,c),g in list(df.groupby(["Part","Machine","Characteristic"]))[:12]:
        with st.expander(f"{p} / {m} / {c}", expanded=df[["Part","Machine","Characteristic"]].drop_duplicates().shape[0] <= 3):
            st.dataframe(g.sort_values("Timestamp",ascending=False),use_container_width=True,hide_index=True)


def render_combination(df):
    if df.empty: st.warning("No data matched that request."); return
    d=df.sort_values("Timestamp").copy(); d["Process"]=process_name(d)
    daily=d.set_index("Timestamp").groupby("Process")["Value"].resample("D").agg(["mean","std"]).reset_index()
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    for name,g in daily.groupby("Process"):
        fig.add_trace(go.Scatter(x=g["Timestamp"],y=g["mean"],name=f"{name} mean"),secondary_y=False)
        fig.add_trace(go.Bar(x=g["Timestamp"],y=g["std"],name=f"{name} std",opacity=.3),secondary_y=True)
    fig.update_layout(title="Combination Chart: Mean and Variation"); show_chart(fig,"combo")


CHARTS={
    "control":render_control,"histogram":render_histogram,"statistics":render_statistics,"data_table":render_data_table,
    "combination":render_combination,"xbar_r":render_xbar_r,"xbar_s":render_xbar_s,"individual_mr":render_imr,
    "scatter":render_scatter,"trend":render_trend_multi,"ewma":render_ewma,"moving_average":render_moving_average,
    "time_axis":render_time_axis,"cusum":render_cusum,"normal_probability":render_normal_probability,
    "performance":render_performance,"capability":render_capability,"sparkline":render_sparkline,
    "combined_control":render_combined_control,"monitor":render_monitor,"multiple_data_table":render_multiple_tables,
}


def render_compare_v2(df, dimension=None, chart=None):
    if df.empty: st.warning("No data matched that comparison."); return
    if not dimension or df[dimension].nunique()<2:
        dimension=next((c for c in ["Machine","Part","Characteristic"] if df[c].nunique()>1),"Machine")
    st.subheader(f"Comparison by {dimension}")
    summary=df.groupby(dimension,as_index=False).agg(Mean=("Value","mean"),StdDev=("Value","std"),Minimum=("Value","min"),Maximum=("Value","max"),Measurements=("Value","size"),OOS_Rate=("Status",lambda s:(s=="Out of Spec").mean()*100))
    if chart and chart in CHARTS and chart not in ["statistics","data_table","monitor","multiple_data_table"]:
        CHARTS[chart](df)
    else:
        fig=px.box(df,x=dimension,y="Value",color=dimension,points="outliers",facet_col="Characteristic" if dimension!="Characteristic" and df["Characteristic"].nunique()>1 else None,title=f"Comparison by {dimension}")
        show_chart(fig,"compare")
    st.dataframe(summary.round(4),use_container_width=True,hide_index=True)
    st.dataframe(calculate_capability_table(df).round(4),use_container_width=True,hide_index=True)


def render_dashboard(df):
    if df.empty: st.warning("No data matched that dashboard request."); return
    t=calculate_capability_table(df); c1,c2,c3,c4=st.columns(4)
    c1.metric("Measurements",f"{len(df):,}"); c2.metric("Processes",len(t)); c3.metric("Lowest Cpk","N/A" if t.empty or pd.isna(t["Cpk"].min()) else f"{t['Cpk'].min():.2f}"); c4.metric("OOS Rate",f"{(df['Status']=='Out of Spec').mean()*100:.2f}%")
    left,right=st.columns(2)
    with left: render_trend_multi(df)
    with right:
        if not t.empty:
            p=t.copy(); p["Process"]=p["Part"]+" / "+p["Machine"]+" / "+p["Characteristic"]
            fig=px.bar(p.sort_values("Cpk").head(12),x="Process",y="Cpk",title="Capability Ranking"); fig.add_hline(y=1.33,line_dash="dash"); show_chart(fig,"dashcap")
    left2,right2=st.columns(2)
    with left2:
        o=df.assign(OOS=(df["Status"]=="Out of Spec").astype(int)).groupby(["Machine","Part"],as_index=False).agg(Measurements=("Value","size"),OOS_Count=("OOS","sum")); o["OOS Rate %"]=o["OOS_Count"]/o["Measurements"]*100
        show_chart(px.bar(o,x="Machine",y="OOS Rate %",color="Part",barmode="group",title="OOS Rate by Machine and Part"),"dashoos")
    with right2:
        latest=df.sort_values("Timestamp").groupby(["Part","Machine","Characteristic"],as_index=False).tail(1).copy(); latest["Process"]=process_name(latest)
        show_chart(px.bar(latest,x="Process",y="Value",color="Status",title="Latest Measurements"),"dashlatest")
    st.dataframe(t.round(4),use_container_width=True,hide_index=True)


st.title("🏭 Manufacturing Copilot")
st.caption("Ask for any chart, compare parts/machines/traces, or ask the copilot to build a dashboard.")

with st.sidebar:
    st.header("Data")
    source=st.radio("Choose data source",["Demo data","Upload CSV or Excel"])
    if source=="Upload CSV or Excel":
        uploaded_file=st.file_uploader("Upload manufacturing data",type=["csv","xlsx","xls"])
        if uploaded_file is None: st.info("Upload a file or switch to Demo data."); st.stop()
        try: data=load_uploaded_data(uploaded_file)
        except Exception as exc: st.error(str(exc)); st.stop()
    else: data=generate_demo_data()
    min_date=data["Timestamp"].min().date(); max_date=data["Timestamp"].max().date(); default_start=max(min_date,max_date-timedelta(days=30))
    selected_dates=st.date_input("Date range",value=(default_start,max_date),min_value=min_date,max_value=max_date)
    if isinstance(selected_dates,tuple) and len(selected_dates)==2: start_date,end_date=selected_dates
    else: start_date=end_date=selected_dates
    st.header("Optional filters")
    part_filter=st.selectbox("Part",["All",*sorted(data["Part"].unique())])
    machine_filter=st.selectbox("Machine",["All",*sorted(data["Machine"].unique())])
    characteristic_filter=st.selectbox("Trace / Characteristic",["All",*sorted(data["Characteristic"].unique())])
    st.download_button("Download current data",data=data.to_csv(index=False).encode("utf-8"),file_name="manufacturing_copilot_data.csv",mime="text/csv")

base_filtered=filter_data_v2(data,[] if part_filter=="All" else [part_filter],[] if machine_filter=="All" else [machine_filter],[] if characteristic_filter=="All" else [characteristic_filter],start_date,end_date)

with st.expander("Example questions",expanded=True):
    st.markdown("""
- Show a histogram for P1 on M2 Length
- Create an X-Bar and R chart for P2 Weight
- Show an EWMA chart for M2 Length
- Show a CUSUM chart for P1 M2 Length
- Show a normal probability plot for Thickness
- Compare M1 vs M2 for P2 Length
- Compare P1 and P2 using a trend chart
- Compare Length, Weight, and Thickness for M2
- Create a dashboard for P1
- Build a dashboard comparing P1 and P2
- Show a monitor table
- Show the raw data table for P1
""")

if "messages" not in st.session_state: st.session_state.messages=[]
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

question=st.chat_input("Ask for a chart, comparison, dashboard, statistics, or data table")
if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)
    parsed=parse_question_v2(question,base_filtered if not base_filtered.empty else data)
    qdata=filter_data_v2(data,parsed["parts"] or ([] if part_filter=="All" else [part_filter]),parsed["machines"] or ([] if machine_filter=="All" else [machine_filter]),parsed["traces"] or ([] if characteristic_filter=="All" else [characteristic_filter]),start_date,end_date)
    labels={"help":"Here are examples of supported requests.","dashboard":"I created a dashboard from the matching data.","compare":"I compared the matching parts, machines, or traces.","chart":f"I created the {str(parsed['chart']).replace('_',' ')}.","nelson":"I checked Nelson Rules 1, 2, and 3.","worst_cpk":"I ranked the processes with the lowest Cpk.","best_cpk":"I ranked the processes with the highest Cpk.","capability":"I calculated process capability.","oos":"I summarized out-of-spec performance.","summary":"I created a manufacturing quality summary."}
    response=labels[parsed["intent"]]; st.session_state.messages.append({"role":"assistant","content":response})
    with st.chat_message("assistant"):
        st.markdown(response); intent=parsed["intent"]
        if intent=="help": st.markdown("Ask for a chart by name, compare parts/machines/traces, or say **create a dashboard**.")
        elif intent=="dashboard": render_dashboard(qdata)
        elif intent=="compare": render_compare_v2(qdata,parsed["compare_dimension"],parsed["chart"])
        elif intent=="chart": CHARTS[parsed["chart"]](qdata)
        elif intent in ["worst_cpk","best_cpk"]:
            t=calculate_capability_table(qdata)
            if t.empty: st.warning("No data matched that request.")
            else:
                asc=intent=="worst_cpk"; ranked=t.sort_values("Cpk",ascending=asc).head(int(parsed["limit"])); keys=ranked.set_index(["Part","Machine","Characteristic"]).index; chosen=qdata[qdata.set_index(["Part","Machine","Characteristic"]).index.isin(keys)]; render_capability(chosen,"Worst Cpk Processes" if asc else "Best Cpk Processes")
        elif intent=="capability": render_capability(qdata)
        elif intent=="nelson": render_nelson(qdata)
        elif intent=="oos": render_oos(qdata)
        else: render_dashboard(qdata)

st.divider(); st.caption("Controlled natural-language parser; no unrestricted SQL is generated or executed.")
