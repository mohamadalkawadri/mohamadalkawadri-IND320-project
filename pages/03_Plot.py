import streamlit as st
import pandas as pd
import altair as alt
from utils import load_data, mask_month_range

st.set_page_config(page_title="Plot")

st.title("Interactive Plot")

df = load_data()

num_cols = df.select_dtypes(include="number").columns.tolist()

# --- Controls ---
left, right = st.columns([1, 2])

with left:
    column_choice = st.selectbox(
        "Select column(s) to plot",
        options=["All columns"] + num_cols,
        index=0,
    )

with right:
    global month_range
    month_range = st.select_slider(
        "Select month range (inclusive)",
        options=range(1, 13),
        value=(1, 2),
    )

mask = mask_month_range(df, month_range[0], month_range[1])
df_masked = df.loc[mask].copy()
plot_df = df.loc[mask, num_cols].copy()

# Prepare data for Altair (wide → long)
plot_df["__time__"] = df_masked["time"]
long_df = plot_df.melt(
    id_vars="__time__",
    value_vars=num_cols,
    var_name="Variable",
    value_name="Value",
)

if column_choice != "All columns":
    long_df = long_df[long_df["Variable"] == column_choice]

x_type = "temporal"

base = alt.Chart(long_df).mark_line().encode(
    x=alt.X("__time__", type=x_type, title="Time"),
    y=alt.Y("Value:Q", title="Value"),
    color=alt.Color("Variable:N", title="Series"),
    tooltip=[
        alt.Tooltip("__time__", type=x_type, title="Time"),
        alt.Tooltip("Variable:N"),
        alt.Tooltip("Value:Q", format=".3f"),
    ],
).properties(
    width="container",
    height=420,
    title=(
        f"Open-Meteo Data — {column_choice}"
        if column_choice != "All columns"
        else "Open-Meteo Data — All Columns"
    ),
)

st.altair_chart(base, use_container_width=True)