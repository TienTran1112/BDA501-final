import datetime as dt

import altair as alt
import streamlit as st

import charts
import serving

catalog = [doc for doc in serving.catalog() if doc.get("combos")]
by_id = {doc["_id"]: doc for doc in catalog}
st.caption("Cheapest hotels for a destination and date, compared side by side, with the full price history of any hotel.")

with st.container(horizontal=True):
    destination = st.selectbox("Destination", list(by_id), format_func=lambda d: f"{d}  ({by_id[d]['rows']:,} price rows)", key="monitor_destination")
    combos = by_id[destination]["combos"]
    busiest = max(combos, key=lambda c: c["hotels"])
    checkins = sorted({c["checkin_date"] for c in combos})
    checkin = st.selectbox("Check-in date", checkins, index=checkins.index(busiest["checkin_date"]), key=f"monitor_checkin_{destination}")
    snapshots = {c["snapshot_date"]: c["hotels"] for c in combos if c["checkin_date"] == checkin}
    ordered = sorted(snapshots)
    snapshot = st.selectbox("Price seen on", ordered, index=ordered.index(max(snapshots, key=snapshots.get)),
                            format_func=lambda s: f"{s}  ({snapshots[s]} hotels)", key=f"monitor_snapshot_{destination}_{checkin}")

cheapest, q2_ms = serving.cheapest(destination, dt.date.fromisoformat(checkin), dt.date.fromisoformat(snapshot))
details, q3_ms = serving.profiles(cheapest["hotel_id"].tolist())
table = cheapest.merge(details, how="left", on="hotel_id")
table.insert(0, "rank", range(1, len(table) + 1))

with st.container(horizontal=True):
    st.metric("Cheapest price", f"${table['price'].iloc[0]:,.2f}", border=True)
    st.metric(f"Median of the {len(table)} cheapest", f"${table['price'].median():,.2f}", border=True)
    st.metric("Hotels with a price that day", f"{snapshots[snapshot]:,}", border=True)
    st.metric("Cassandra query (Q2)", f"{q2_ms:.1f} ms", border=True)
    st.metric("MongoDB profiles (Q3)", f"{q3_ms:.1f} ms", border=True)

with st.container(border=True):
    st.subheader("Hotel comparison")
    st.caption("Sorted by price, cheapest first. Select a row to see that hotel's price history below.")
    event = st.dataframe(
        table[["rank", "hotel_id", "price", "stars", "review_score", "brand", "lowest_30d", "highest_30d"]],
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"monitor_table_{destination}_{checkin}_{snapshot}",
        alt="The 20 cheapest hotels for the chosen destination and dates",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "hotel_id": st.column_config.NumberColumn("Hotel ID", format="%d"),
            "price": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
            "stars": st.column_config.NumberColumn("Stars", format="%.1f"),
            "review_score": st.column_config.NumberColumn("Review score of 5", format="%.1f"),
            "brand": st.column_config.CheckboxColumn("Chain hotel"),
            "lowest_30d": st.column_config.NumberColumn("Lowest, June 2013", format="$%.2f"),
            "highest_30d": st.column_config.NumberColumn("Highest, June 2013", format="$%.2f"),
        },
    )

long_histories = sorted(by_id[destination].get("long_histories", []), key=lambda h: (-h["rows"], h["hotel_id"], h["checkin_date"]))
options = [(h["hotel_id"], h["checkin_date"], h["rows"]) for h in long_histories]
picked = None
if event.selection.rows:
    picked = (int(table["hotel_id"].iloc[event.selection.rows[0]]), checkin, None)
    options = [picked] + [o for o in options if o[:2] != picked[:2]]

with st.container(border=True):
    st.subheader("Price history of one hotel")
    if not options:
        st.info("Select a hotel in the table above.", icon=":material/touch_app:")
        st.stop()
    hotel, history_checkin, _ = st.selectbox(
        "Hotel and check-in date", options, key=f"monitor_history_{destination}_{picked}",
        format_func=lambda o: f"Hotel {o[0]}, check-in {o[1]}" + (f"  ({o[2]} recorded prices)" if o[2] else "  (selected in the table)"))
    history, q1_ms = serving.price_history(hotel, dt.date.fromisoformat(history_checkin))
    profile, _ = serving.profiles([hotel])

    left, right = st.columns([3, 1])
    with right:
        doc = profile.iloc[0]
        def known(value, pattern):
            return "Not recorded" if value is None or value != value else pattern.format(value)
        st.table({
            "Stars": known(doc["stars"], "{:.1f}"),
            "Review score": known(doc["review_score"], "{:.1f} of 5"),
            "Chain hotel": "Yes" if doc["brand"] else "No",
            "Country ID": f"{doc['country_id']:.0f}",
            "Main destination": f"{doc['destination_id']:.0f}",
            "June 2013 range": "No prices" if doc["lowest_30d"] != doc["lowest_30d"] else f"\\${doc['lowest_30d']:,.2f} to \\${doc['highest_30d']:,.2f}",
            "Cassandra query (Q1)": f"{q1_ms:.1f} ms",
        }, border="horizontal", alt="Hotel profile")
    with left:
        if len(history) >= 2:
            first, last = history["price"].iloc[0], history["price"].iloc[-1]
            with st.container(horizontal=True):
                st.metric("Recorded prices", f"{len(history):,}", border=True)
                st.metric("Lowest", f"${history['price'].min():,.2f}", border=True)
                st.metric("Highest", f"${history['price'].max():,.2f}", border=True)
                st.metric("Latest", f"${last:,.2f}", f"{last / first - 1:+.1%} since first price", delta_color="off", border=True)
            st.altair_chart(charts.trend_line(history, "snapshot_ts", "price", "Price seen on", "Price (USD)", zero=False, points=True),
                            alt=f"Price history of hotel {hotel} for check-in {history_checkin}")
        else:
            st.info("Only one price was recorded for this hotel and check-in date, so there is no trend to draw.", icon=":material/show_chart:")
        with st.expander("Show the prices as a table"):
            st.dataframe(history, hide_index=True, alt="Recorded prices", column_config={
                "snapshot_ts": st.column_config.DatetimeColumn("Price seen on", format="YYYY-MM-DD HH:mm"),
                "price": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
            })
