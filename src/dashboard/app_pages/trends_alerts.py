import altair as alt
import pandas as pd
import streamlit as st

import charts
import serving

by_id = {doc["_id"]: doc for doc in serving.catalog()}
st.caption("Weekly median price and promotion rate for a destination, plus unusual price rises across all destinations.")

if st.session_state.get("trend_destination") not in by_id:
    st.session_state.trend_destination = next(iter(by_id))
destination = st.selectbox("Destination", list(by_id), format_func=lambda d: f"{d}  ({by_id[d]['rows']:,} price rows)", key="trend_destination")
doc = by_id[destination]

weekly, q5_ms = serving.weekly(destination)
alerts = serving.alerts_for(destination)
total_alerts, alerted_destinations = serving.alert_totals()

latest = weekly.iloc[-1]
previous = weekly.iloc[-2] if len(weekly) > 1 else None
with st.container(horizontal=True):
    st.metric(f"Median price, week of {latest['week_start']:%d %b %Y}", f"${latest['median_price']:,.2f}",
              f"{latest['median_price'] / previous['median_price'] - 1:+.1%} on previous week" if previous is not None else None,
              delta_color="off", border=True)
    st.metric("Promotion rate that week", f"{latest['promotion_rate']:.1%}", border=True)
    st.metric("Alert weeks here", f"{len(alerts):,}", border=True)
    st.metric("MongoDB query (Q5)", f"{q5_ms:.1f} ms", border=True)

extra = [alt.Tooltip("searches_rows:Q", title="Price rows", format=",")]
markers = alerts.assign(label=lambda f: [f"+{r - 1:.0%}" for r in f["ratio"]])
left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Median price per week")
    st.altair_chart(charts.trend_line(weekly, "week_start", "median_price", "Week", "Median price (USD)", extra_tooltips=extra, markers=markers),
                    alt=f"Weekly median price for destination {destination}")
    if not alerts.empty:
        st.caption(":material/change_history: Orange triangle: price alert, labelled with the rise over the previous four weeks.")
with right, st.container(border=True):
    st.subheader("Promotion rate per week")
    st.altair_chart(charts.trend_line(weekly, "week_start", "promotion_rate", "Week", "Share of prices on promotion", y_format=".0%", extra_tooltips=extra, color="series2"),
                    alt=f"Weekly promotion rate for destination {destination}")

with st.expander("Show the weekly figures as a table"):
    st.dataframe(weekly, hide_index=True, alt="Weekly price and promotion figures", column_config={
        "week_start": st.column_config.DateColumn("Week starting"),
        "searches_rows": st.column_config.NumberColumn("Price rows", format="localized"),
        "promotion_rate": st.column_config.NumberColumn("Promotion rate", format="percent"),
        "median_price": st.column_config.NumberColumn("Median price", format="$%.2f"),
        "mean_price": st.column_config.NumberColumn("Mean price", format="$%.2f"),
    })

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Median price by star rating")
    stars = pd.DataFrame(doc.get("by_star", []))
    if stars.empty:
        st.info("Too few rows for this destination.", icon=":material/info:")
    else:
        st.altair_chart(charts.star_bars(stars), alt=f"Median price by star rating for destination {destination}")
with right, st.container(border=True):
    st.subheader("Median price by check-in week")
    by_checkin = pd.DataFrame(doc.get("by_checkin_week", []))
    if by_checkin.empty:
        st.info("Too few rows for this destination.", icon=":material/info:")
    else:
        by_checkin["week"] = pd.to_datetime(by_checkin["week"])
        st.altair_chart(charts.trend_line(by_checkin, "week", "median_price", "Check-in week", "Median price (USD)",
                                          extra_tooltips=[alt.Tooltip("rows:Q", title="Price rows", format=",")], height=max(260, 44 * len(doc.get("by_star", [])))),
                        alt=f"Median price by check-in week for destination {destination}")

top, _ = serving.top_alerts()
top["status"] = "▲ Price spike"
top["rise"] = (top["ratio"] - 1) * 100


def open_alert():
    rows = st.session_state.alert_table.selection.rows
    if rows and int(top["destination_id"].iloc[rows[0]]) in by_id:
        st.session_state.trend_destination = int(top["destination_id"].iloc[rows[0]])


with st.container(border=True):
    st.subheader("Price alerts across all destinations")
    st.caption(f"{total_alerts:,} alert weeks in {alerted_destinations:,} destinations, largest rise first (top {len(top)} shown). Select a row to open that destination above.")
    st.markdown("**Rule:** the week's median is at least 25% above the median of the previous four weekly medians. The week and at least three of those four "
                "must each have 200 or more price rows. **Read with care:** the hotels and travel dates people searched change from week to week, so an alert may reflect a change in what was searched and not a price rise.")
    st.dataframe(top[["status", "week_start", "destination_id", "median_price", "trailing_median", "rise", "searches_rows"]], hide_index=True,
                 on_select=open_alert, selection_mode="single-row", key="alert_table", alt="Largest weekly price rises across destinations", column_config={
        "status": st.column_config.TextColumn("Status"),
        "week_start": st.column_config.DateColumn("Week starting"),
        "destination_id": st.column_config.NumberColumn("Destination", format="%d"),
        "median_price": st.column_config.NumberColumn("Median that week", format="$%.2f"),
        "trailing_median": st.column_config.NumberColumn("Previous four weeks", format="$%.2f"),
        "rise": st.column_config.NumberColumn("Rise", format="+%.0f%%"),
        "searches_rows": st.column_config.NumberColumn("Price rows", format="localized"),
    })
