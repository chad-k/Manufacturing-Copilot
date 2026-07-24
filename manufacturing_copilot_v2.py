import re
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Manufacturing Copilot Prototype",
    page_icon="🏭",
    layout="wide",
)


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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(grouped.round(3), use_container_width=True, hide_index=True)


def render_summary(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data matched that request.")
        return
    st.write("Here is a manufacturing quality summary for the selected period.")
    render_capability(df, "Capability Overview")


st.title("🏭 Manufacturing Copilot Prototype")
st.caption("Ask manufacturing questions in plain English. This prototype uses a controlled parser, so no API key is required.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Choose data source", ["Demo data", "Upload CSV or Excel"])

    if source == "Upload CSV or Excel":
        uploaded_file = st.file_uploader("Upload manufacturing data", type=["csv", "xlsx", "xls"])
        if uploaded_file is None:
            st.info("Upload a file or switch to Demo data.")
            st.stop()
        try:
            data = load_uploaded_data(uploaded_file)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    else:
        data = generate_demo_data()

    min_date = data["Timestamp"].min().date()
    max_date = data["Timestamp"].max().date()
    default_start = max(min_date, max_date - timedelta(days=30))
    selected_dates = st.date_input(
        "Date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    st.header("Optional filters")
    part_filter = st.selectbox("Part", ["All", *sorted(data["Part"].unique())])
    machine_filter = st.selectbox("Machine", ["All", *sorted(data["Machine"].unique())])
    characteristic_filter = st.selectbox(
        "Characteristic", ["All", *sorted(data["Characteristic"].unique())]
    )

    st.download_button(
        "Download demo/current data",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="manufacturing_copilot_data.csv",
        mime="text/csv",
    )

base_filtered = filter_data(
    data,
    None if part_filter == "All" else part_filter,
    None if machine_filter == "All" else machine_filter,
    None if characteristic_filter == "All" else characteristic_filter,
    start_date,
    end_date,
)

with st.expander("Example questions", expanded=True):
    st.markdown(
        """
- Show the five worst Cpk values
- Show Cpk for P1 on M2 for Length
- Show the trend for P1 on M2 Length
- Compare M1 and M2 for P2
- Show Nelson violations for P1 on M2
- Which process has the most out-of-spec measurements?
- Give me a quality summary
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about Cpk, trends, Nelson rules, comparisons, or out-of-spec results")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    parsed = parse_question(question, base_filtered if not base_filtered.empty else data)

    query_data = filter_data(
        data,
        parsed["part"] or (None if part_filter == "All" else part_filter),
        parsed["machine"] or (None if machine_filter == "All" else machine_filter),
        parsed["characteristic"] or (None if characteristic_filter == "All" else characteristic_filter),
        start_date,
        end_date,
    )

    intent = parsed["intent"]
    response_text = {
        "help": "Here are the questions this prototype understands.",
        "worst_cpk": "I ranked the processes with the lowest Cpk.",
        "best_cpk": "I ranked the processes with the highest Cpk.",
        "capability": "I calculated capability for the matching process data.",
        "trend": "I created a time-series view with specification and control references.",
        "nelson": "I checked Nelson Rules 1, 2, and 3.",
        "compare": "I compared the matching manufacturing processes.",
        "oos": "I summarized out-of-spec performance.",
        "summary": "I generated a quality and capability summary.",
    }[intent]

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

        if intent == "help":
            st.markdown(
                "Ask about **Cpk**, **capability**, **trends**, **Nelson violations**, **machine comparisons**, or **out-of-spec results**. Include a part, machine, or characteristic to narrow the result."
            )
        elif intent in {"worst_cpk", "best_cpk"}:
            table = calculate_capability_table(query_data)
            if table.empty:
                st.warning("No data matched that request.")
            else:
                ascending = intent == "worst_cpk"
                ranked = table.sort_values("Cpk", ascending=ascending).head(int(parsed["limit"]))
                render_capability(
                    query_data[
                        query_data.set_index(["Part", "Machine", "Characteristic"]).index.isin(
                            ranked.set_index(["Part", "Machine", "Characteristic"]).index
                        )
                    ],
                    "Worst Cpk Processes" if ascending else "Best Cpk Processes",
                )
        elif intent == "capability":
            render_capability(query_data)
        elif intent == "trend":
            render_trend(query_data)
        elif intent == "nelson":
            render_nelson(query_data)
        elif intent == "compare":
            render_compare(query_data)
        elif intent == "oos":
            render_oos(query_data)
        else:
            render_summary(query_data)

st.divider()
st.caption(
    "Prototype note: this version uses approved calculations and a controlled natural-language parser. It does not generate or execute unrestricted SQL."
)
