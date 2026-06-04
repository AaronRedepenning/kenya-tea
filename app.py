from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

full_df = pd.read_csv("data/tbk-performance.csv")
full_df["date"] = pd.to_datetime(full_df["date"])
full_df["value"] /= 1_000_000

st.set_page_config(
    # Title and icon for the browser's tab bar:
    page_title="Kenya Tea Dataset",
    page_icon="🍵",
    # Make the content take up the width of the page:
    layout="wide",
)


"""
# Kenya Tea Dataset

For exploring data from the [Tea Board of Kenya](https://www.teaboard.or.ke/)!
"""

"""
## Compare different years
"""

YEARS = full_df["date"].dt.year.unique()
selected_years = st.pills(
    "Years to compare", YEARS, default=YEARS, selection_mode="multi"
)

if not selected_years:
    st.warning("You must select at least 1 year.", icon=":material/warning:")

df = full_df[full_df["date"].dt.year.isin(selected_years)]

cols = st.columns([3, 1])

with cols[0].container(border=True, height="stretch"):
    st.markdown("### Tea Production")

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", timeUnit="monthdate", title="Date"),
            y=alt.Y("value:Q", title="Tea production (million kg)"),
            color=alt.Color("year(date):N", title="Year"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("value:Q", title="Tea production", format=",.0f"),
            ],
        )
        .configure_legend(orient="bottom")
    )

    st.altair_chart(chart)

with cols[0].container(border=True, height="stretch"):
    "### Raw data"

    st.dataframe(df)
