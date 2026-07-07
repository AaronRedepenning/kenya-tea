from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

DATA_FILE = "data/tbk_dataset.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce") / 1_000_000

    df = df.dropna(subset=["date", "value"]).copy()
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()

    return df.sort_values(["date", "region"]).reset_index(drop=True)


st.set_page_config(
    page_title="Kenya Tea Dataset",
    page_icon="🍵",
    layout="wide",
)


full_df = load_data()


"""
# Kenya Tea Dataset

For exploring data from the [Tea Board of Kenya](https://www.teaboard.or.ke/)!
"""


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------

all_years = sorted(full_df["year"].dropna().unique())

selected_years = st.pills(
    "Years to compare",
    all_years,
    default=all_years,
    selection_mode="multi",
)

if not selected_years:
    st.warning("You must select at least 1 year.", icon=":material/warning:")
    st.stop()


# -----------------------------------------------------------------------------
# Chart 1: Monthly seasonal overlay
# -----------------------------------------------------------------------------

"""
## Compare different years
"""

seasonal_region = st.selectbox(
    "Region for seasonal comparison",
    options=["all", "west_of_rift", "east_of_rift"],
    format_func=lambda x: {
        "all": "All",
        "west_of_rift": "West of Rift",
        "east_of_rift": "East of Rift",
    }.get(x, x),
)

seasonal_df = full_df[
    (full_df["region"] == seasonal_region) & (full_df["year"].isin(selected_years))
].copy()

cols = st.columns([3, 1])

with cols[0].container(border=True):
    st.markdown("### Monthly Tea Production by Year")

    seasonal_chart = (
        alt.Chart(seasonal_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month_num:O",
                title="Month",
                sort=list(range(1, 13)),
                axis=alt.Axis(
                    labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value - 1]"
                ),
            ),
            y=alt.Y("value:Q", title="Tea production (million kg)"),
            color=alt.Color("year:N", title="Year"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %Y"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("value:Q", title="Tea production", format=",.1f"),
            ],
        )
        .properties(height=430)
        .configure_legend(orient="bottom")
    )

    st.altair_chart(seasonal_chart, use_container_width=True)

with cols[1].container(border=True):
    st.markdown("### Current selection")

    st.metric("Rows", f"{len(seasonal_df):,}")

    if not seasonal_df.empty:
        st.metric(
            "Average production",
            f"{seasonal_df['value'].mean():,.1f}M kg",
        )
        st.metric(
            "Max production",
            f"{seasonal_df['value'].max():,.1f}M kg",
        )


# -----------------------------------------------------------------------------
# Chart 2: Components over time
# -----------------------------------------------------------------------------

"""
## Regional components over time
"""

component_regions = ["all", "west_of_rift", "east_of_rift"]

components_df = full_df[
    full_df["region"].isin(component_regions) & full_df["year"].isin(selected_years)
].copy()

region_label_map = {
    "all": "All",
    "west_of_rift": "West of Rift",
    "east_of_rift": "East of Rift",
}

components_df["region_label"] = components_df["region"].map(region_label_map)

with st.container(border=True):
    st.markdown("### All vs. West of Rift vs. East of Rift")

    components_chart = (
        alt.Chart(components_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Tea production (million kg)"),
            color=alt.Color(
                "region_label:N",
                title="Region",
                sort=["All", "West of Rift", "East of Rift"],
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %Y"),
                alt.Tooltip("region_label:N", title="Region"),
                alt.Tooltip("value:Q", title="Tea production", format=",.1f"),
            ],
        )
        .properties(height=430)
        .configure_legend(orient="bottom")
    )

    st.altair_chart(components_chart, use_container_width=True)


# -----------------------------------------------------------------------------
# Raw data
# -----------------------------------------------------------------------------

"""
## Raw data
"""

display_regions = st.multiselect(
    "Regions to show",
    options=component_regions,
    default=component_regions,
    format_func=lambda x: region_label_map.get(x, x),
)

raw_df = full_df[
    full_df["region"].isin(display_regions) & full_df["year"].isin(selected_years)
].copy()

raw_df = raw_df.sort_values(["date", "region"]).reset_index(drop=True)

with st.container(border=True):
    st.dataframe(
        raw_df[
            [
                "date",
                "year",
                "month",
                "metric",
                "region",
                "value",
                "unit",
                "source",
                "source_type",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
