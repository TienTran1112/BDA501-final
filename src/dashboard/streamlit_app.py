"""Hotel price dashboard: the last layer of the BDA501 pipeline, reading live from Cassandra and MongoDB."""
import streamlit as st

import serving

st.set_page_config(page_title="Hotel price dashboard", page_icon=":material/hotel:", layout="wide")

try:
    serving.cassandra()
    serving.mongo()
except Exception as error:
    st.error("The dashboard cannot reach the serving databases. Start them with "
             "`docker compose --profile cassandra --profile mongo up -d`, wait until all six are healthy, then reload.",
             icon=":material/database_off:")
    st.caption(f"{type(error).__name__}: {str(error)[:300]}")
    st.stop()

page = st.navigation([
    st.Page("app_pages/price_monitor.py", title="Price monitor", icon=":material/payments:", default=True),
    st.Page("app_pages/trends_alerts.py", title="Trends and alerts", icon=":material/monitoring:"),
    st.Page("app_pages/nearby.py", title="Nearby hotels", icon=":material/near_me:"),
], position="top")

st.title(page.title)
page.run()

st.divider()
with st.expander("What this data can and cannot show", icon=":material/info:"):
    st.markdown("""
- **Source:** Expedia hotel searches from November 2012 to June 2013. This is a historical test of the pipeline, not current prices.
- **Price:** Expedia's displayed price in USD. Expedia notes it can be per night or for the whole stay, depending on the country. Only prices of zero or less and above 5,000 USD were removed, so implausible single prices below 1 USD remain and can appear as a hotel's lowest price.
- **Destination:** Expedia's anonymised search destination number. The data has no city or district names and no hotel names.
- **Sold-out rate:** not available, because Expedia does not record it. Promotion rate is shown in its place.
- **Location:** Expedia publishes no hotel coordinates. The Nearby page uses synthetic ones.
- **Review score:** Expedia's 0 to 5 scale. The proposal's score of 8 out of 10 corresponds to 4 out of 5.
""")
