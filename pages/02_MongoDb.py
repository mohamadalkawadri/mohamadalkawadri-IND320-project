from pymongo.mongo_client import MongoClient
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from bson.son import SON
import plotly.graph_objects as go


st.set_page_config(page_title="MongoDB")

st.title("Page 2 — MongoDB")

@st.cache_resource(show_spinner=False)
def get_mongo_collection():
    uri = st.secrets.get("MONGO_URI")
    client = MongoClient(uri, tlsAllowInvalidCertificates=True)
    database = client['assignment4']
    collection = database['elhub']
    return collection

coll = get_mongo_collection()
year = st.slider(
    "Select year",
    min_value=2021,
    max_value=2024,
    value=2021,
    step=1
)

@st.cache_data(show_spinner=False)
def distinct_price_areas():
    return sorted(list(coll.distinct("priceArea")))

@st.cache_data(show_spinner=False)
def groups_for_area(area: str):
    pipeline = [
        {"$match": {"priceArea": area}},
        {"$group": {"_id": "$productionGroup"}},
        {"$sort": SON([("_id", 1)])}
    ]
    return [d["_id"] for d in coll.aggregate(pipeline)]

@st.cache_data(show_spinner=True)
def load_year(area: str, year: int) -> pd.DataFrame:
    # Pull only what we need
    cur = coll.find(
        {
            "priceArea": area,
            "startTime": {
                "$gte": datetime(year, 1, 1),
                "$lt": datetime(year + 1, 1, 1)
            }
        },
        {
            "_id": 0,
            "priceArea": 1,
            "productionGroup": 1,
            "startTime": 1,
            "quantityKwh": 1
        }
    )
    df = pd.DataFrame(list(cur))
    if not df.empty:
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce")
    return df

@st.cache_data(show_spinner=True)
def load_month(area: str, year: int, month: int) -> pd.DataFrame:
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), (month % 12) + 1, 1)
    cur = coll.find(
        {
            "priceArea": area,
            "startTime": {"$gte": start, "$lt": end}
        },
        {"_id": 0, "priceArea": 1, "productionGroup": 1, "startTime": 1, "quantityKwh": 1}
    )
    df = pd.DataFrame(list(cur))
    if not df.empty:
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce")
    return df

# ── Layout: two columns ────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

# LEFT: price area + PIE (total over the year, grouped by productionGroup)
with left:
    st.subheader("Total Production — Pie")
    areas = distinct_price_areas()
    area_sel = st.radio("Price area", options=areas, index=areas.index(st.session_state.selected_area), horizontal=True)
    year_df = load_year(area_sel, year)
    st.session_state.selected_area = area_sel
    
    if year_df.empty:
        st.info(f"No data for {area_sel} in {year}.")
    else:
        pie_df = (year_df
                  .groupby("productionGroup", as_index=False)["quantityKwh"].sum()
                  .sort_values("quantityKwh", ascending=False))
        fig = go.Figure(
            go.Pie(
                labels=pie_df["productionGroup"],
                values=pie_df["quantityKwh"],
                hovertemplate="<b>%{label}</b><br>%{value:.0f} kWh<br>%{percent}",
                textinfo="percent+label",
                hole=0  # set to 0.4 for donut
            )
        )

        fig.update_layout(
            title=f"{year} Total — {area_sel}",
            height=500,
            margin=dict(t=60, b=20, l=20, r=20)
        )

        st.plotly_chart(fig, use_container_width=True)

# RIGHT: pills (production groups) + month selector → LINE plot
with right:
    st.subheader("Monthly Lines by Production Group")
    # Month selector
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_idx = st.selectbox("Select month", options=list(range(1,13)), format_func=lambda i: f"{i:02d} - {month_names[i-1]}")
    # Groups
    all_groups = groups_for_area(area_sel)

    # Try st.pills (Streamlit >=1.39). Fallback to multiselect if not available.
    selected_groups = None
    try:
        selected_groups = st.pills("Production groups (multi-select)", options=all_groups, selection_mode="multi", default=all_groups)
    except Exception:
        selected_groups = st.multiselect("Production groups", options=all_groups, default=all_groups)

    month_df = load_month(area_sel, year, month_idx)
    if month_df.empty:
        st.info(f"No data for {area_sel}, {year}-{month_idx:02d}.")
    else:
        if selected_groups:
            month_df = month_df[month_df["productionGroup"].isin(selected_groups)]
        month_df = month_df.sort_values("startTime")

        if month_df.empty:
            st.info("No data after applying production group filter.")
        else:
            # Pivot for multi-line plot
            wide = (month_df
                    .pivot_table(index="startTime", columns="productionGroup", values="quantityKwh", aggfunc="sum")
                    .fillna(0.0)
                   )
            fig = go.Figure()

            # Add each column as its own line
            for col in wide.columns:
                fig.add_trace(go.Scatter(
                    x=wide.index,
                    y=wide[col],
                    mode="lines",
                    name=col,
                ))

            fig.update_layout(
                title=f"{year}-{month_idx:02d} — {area_sel}",
                xaxis_title="Time",
                yaxis_title="Quantity (kWh)",
                height=600,
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1
                ),
                margin=dict(l=40, r=40, t=60, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

# ── Expander with short source note ────────────────────────────────────────────
with st.expander("Data source"):
    st.markdown(
        """
        The data shown on this page originates from **Elhub** (dataset:
        `PRODUCTION_PER_GROUP_MBA_HOUR`) downloaded from this url: https://api.elhub.no.
        It was ingested into **MongoDB** (via Python MongoDB library) and is queried live here.
        """
    )