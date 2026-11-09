import streamlit as st

import charts
import serving

st.warning("**These locations are synthetic.** Expedia publishes no hotel coordinates, so each hotel was given a made-up position near a made-up destination centre. "
           "This page shows that the MongoDB geospatial query works. It does not show where any hotel really is, so distances here have no real meaning.",
           icon=":material/warning:")

by_id = {doc["_id"]: doc for doc in serving.catalog()}
with st.container(horizontal=True):
    destination = st.selectbox("Destination", list(by_id), format_func=lambda d: f"{d}  ({by_id[d]['hotels']:,} hotels)", key="nearby_destination")
    centres = serving.hotels_in_destination(destination)
    hotel = st.selectbox("Centre hotel", list(centres), format_func=lambda h: f"Hotel {h}", key=f"nearby_hotel_{destination}")
    radius_km = st.slider("Radius (km)", 1, 15, 10, key="nearby_radius")
    min_score = st.slider("Minimum review score of 5", 0.0, 5.0, 4.0, 0.5, key="nearby_score")

st.caption("The proposal's query uses 2 km and a score of 4 of 5 (8 of 10). The default here is 10 km because the synthetic positions are spread thinly: "
           "in the load test, 35.5% of 2 km searches found nothing.")

found, q4_ms = serving.nearby(centres[hotel], radius_km, min_score, 200)
others = found[found["hotel_id"] != hotel]
priced = others["lowest_30d"].dropna()

with st.container(horizontal=True):
    st.metric("Other hotels found", f"{len(others):,}", border=True)
    st.metric("With a June 2013 price", f"{others['lowest_30d'].notna().sum():,}", border=True)
    st.metric("Median of lowest prices", f"${priced.median():,.2f}" if not priced.empty else "No prices", border=True)
    st.metric("MongoDB query (Q4)", f"{q4_ms:.1f} ms", border=True)

if others.empty:
    st.info("No other hotel matches. Try a larger radius or a lower review score.", icon=":material/search_off:")

left, right = st.columns([3, 2])
with left, st.container(border=True):
    st.subheader("Price map (synthetic coordinates)")
    # The colour scale comes from every hotel in the result, so the same hotel keeps its colour within one query.
    every_price = found["lowest_30d"].dropna()
    low, high = (every_price.min(), every_price.max()) if not every_price.empty else (0, 1)
    st.altair_chart(charts.nearby_scatter(found, radius_km, low, max(high, low + 0.01)), width="content",
                    alt=f"Hotels within {radius_km} km of hotel {hotel}, coloured by lowest June 2013 price, on synthetic coordinates")
    st.caption("Cross: centre hotel's position. Dashed circle: search radius. Grey dot: no price in June 2013. No world map is drawn, because the positions are not real.")
with right, st.container(border=True):
    st.subheader("Hotels found")
    st.dataframe(found[["hotel_id", "distance_km", "review_score", "stars", "lowest_30d"]], hide_index=True, alt="Hotels within the chosen radius", column_config={
        "hotel_id": st.column_config.NumberColumn("Hotel ID", format="%d"),
        "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
        "review_score": st.column_config.NumberColumn("Review score of 5", format="%.1f"),
        "stars": st.column_config.NumberColumn("Stars", format="%.1f"),
        "lowest_30d": st.column_config.NumberColumn("Lowest, June 2013", format="$%.2f"),
    })
