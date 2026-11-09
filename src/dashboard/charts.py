"""Altair charts for the dashboard. One axis per chart, one series colour, recessive grid, text in text colours."""
import math

import altair as alt
import pandas as pd
import streamlit as st

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7", "series": "#2a78d6",
         "series2": "#eb6834", "absent": "#c3c2b7", "ramp": ["#6ea6ea", "#2a78d6", "#0d366b"]}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835", "series": "#3987e5",
        "series2": "#d95926", "absent": "#52514e", "ramp": ["#2a78d6", "#7cb1ee", "#cde2fb"]}
ALERT = "#ec835a"  # The status palette's "serious" step: same in both themes, always shown with a triangle and a label.
USD = "$,.2f"
USD_AXIS = "$,.0f"


def tokens():
    return DARK if st.context.theme.type == "dark" else LIGHT


def recessive(chart):
    t = tokens()
    return chart.configure_axis(gridColor=t["grid"], domainColor=t["baseline"], tickColor=t["baseline"])


def trend_line(frame, x, y, x_title, y_title, y_format=USD_AXIS, zero=True, points=False, color="series", extra_tooltips=(),
               markers=None, height=280):
    """A single-series line with a crosshair tooltip. markers (optional) is a frame of flagged points with a label column."""
    t = tokens()
    span_days = (frame[x].max() - frame[x].min()).days
    time_axis = {"format": "%b %Y", "tickCount": "month"} if span_days >= 90 else {"format": "%d %b", "tickCount": 6}
    x_encoding = alt.X(f"{x}:T", title=x_title, axis=alt.Axis(labelOverlap=True, grid=False, **time_axis))
    y_encoding = alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=zero), axis=alt.Axis(format=y_format, tickCount=5))
    tooltip = [alt.Tooltip(f"{x}:T", title=x_title, format="%d %b %Y"),
               alt.Tooltip(f"{y}:Q", title=y_title, format=USD if y_format == USD_AXIS else y_format), *extra_tooltips]
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(frame).encode(x=x_encoding)
    layers = [base.mark_line(strokeWidth=2, color=t[color]).encode(y=y_encoding)]
    if points:
        layers.append(base.mark_point(filled=True, size=72, opacity=1, color=t[color], stroke=t["surface"], strokeWidth=1.5).encode(y=y_encoding))
    layers.append(base.mark_rule(color=t["muted"]).encode(opacity=alt.condition(hover, alt.value(0.7), alt.value(0)), tooltip=tooltip).add_params(hover))
    if markers is not None and not markers.empty:
        marks = alt.Chart(markers).encode(x=alt.X(f"{x}:T"), y=alt.Y(f"{y}:Q"))
        layers.append(marks.mark_point(shape="triangle-up", filled=True, opacity=1, size=150, color=ALERT, stroke=t["surface"], strokeWidth=1.5))
        layers.append(marks.mark_text(dy=-16, fontWeight="bold", color=t["ink"]).encode(text="label:N"))
    return recessive(alt.layer(*layers).properties(height=height))


def star_bars(frame):
    t = tokens()
    y = alt.Y("star:O", title="Star rating", sort="descending", axis=alt.Axis(labelExpr="format(datum.value, '.1f')"))
    x = alt.X("median_price:Q", title="Median price (USD)", axis=alt.Axis(format=USD_AXIS, tickCount=4))
    base = alt.Chart(frame).encode(y=y, x=x, tooltip=[alt.Tooltip("star:O", title="Star rating"), alt.Tooltip("median_price:Q", title="Median price", format=USD),
                                                      alt.Tooltip("rows:Q", title="Price rows", format=",")])
    bars = base.mark_bar(size=16, cornerRadiusEnd=4, color=t["series"])
    labels = base.mark_text(align="left", dx=5, color=t["ink"]).encode(text=alt.Text("median_price:Q", format=USD_AXIS))
    return recessive(alt.layer(bars, labels).properties(height=max(260, 44 * len(frame))))


def nearby_scatter(frame, radius_km, min_price, max_price):
    """Kilometres east and north of the centre hotel. Equal scales, so a circle looks like a circle. No basemap: the coordinates are synthetic."""
    t = tokens()
    edge = radius_km * 1.08
    x = alt.X("east_km:Q", title="km east of the centre hotel (synthetic)", scale=alt.Scale(domain=[-edge, edge]))
    y = alt.Y("north_km:Q", title="km north of the centre hotel (synthetic)", scale=alt.Scale(domain=[-edge, edge]))
    tooltip = [alt.Tooltip("hotel_id:N", title="Hotel"), alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
               alt.Tooltip("review_score:Q", title="Review score of 5", format=".1f"), alt.Tooltip("stars:Q", title="Stars", format=".1f"),
               alt.Tooltip("lowest_30d:Q", title="Lowest price, June 2013", format=USD)]
    ring = pd.DataFrame([{"east_km": radius_km * math.sin(2 * math.pi * i / 96), "north_km": radius_km * math.cos(2 * math.pi * i / 96), "step": i} for i in range(97)])
    layers = [alt.Chart(ring).mark_line(color=t["muted"], strokeWidth=1, strokeDash=[4, 4]).encode(x=x, y=y, order="step:Q")]

    missing = frame[frame["lowest_30d"].isna()]
    if not missing.empty:
        layers.append(alt.Chart(missing).mark_circle(size=90, opacity=1, color=t["absent"], stroke=t["surface"], strokeWidth=1.5).encode(x=x, y=y, tooltip=tooltip))
    priced = frame[frame["lowest_30d"].notna()]
    if not priced.empty:
        scale = alt.Scale(domain=[min_price, max_price], range=t["ramp"], clamp=True)
        legend = alt.Legend(title="Lowest price, June 2013 (USD)", format=USD if max_price - min_price < 10 else USD_AXIS,
                            orient="bottom", gradientLength=240, titleLimit=240)
        layers.append(alt.Chart(priced).mark_circle(size=90, opacity=1, stroke=t["surface"], strokeWidth=1.5)
                      .encode(x=x, y=y, color=alt.Color("lowest_30d:Q", scale=scale, legend=legend), tooltip=tooltip))
    centre = pd.DataFrame([{"east_km": 0.0, "north_km": 0.0}])
    layers.append(alt.Chart(centre).mark_point(shape="cross", filled=True, opacity=1, size=190, color=t["ink"]).encode(x=x, y=y))
    # "pad" makes width and height the plot area itself, so the two scales stay equal whatever the axes and legend need.
    return recessive(alt.layer(*layers).properties(width=480, height=480, autosize=alt.AutoSizeParams(type="pad")))
