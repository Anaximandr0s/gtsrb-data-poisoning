"""Altair charts of the dashboard, built from the saved results only.

Style: thin marks, recessive grid, one colour per entity (model role or detection method),
values in text ink with a tooltip on every mark.
"""

import altair as alt
import numpy as np
import pandas as pd

from src.dashboard import METHOD_LABELS, RATE_STUDY_METHODS, ROLE_LABELS, TSNE_GROUPS, model_role

TEXT, MUTED, GRID = "#191919", "#666666", "#ececec"
PERCENT = ".1%"


def _style(chart: alt.TopLevelMixin, height: int) -> alt.TopLevelMixin:
    """Shared look: transparent view, light grid, muted axis text."""
    return (
        chart.properties(height=height, width="container")
        .configure_view(stroke=None)
        .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=MUTED, titleColor=MUTED,
                        labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
        .configure_legend(labelColor=TEXT, titleColor=MUTED, labelFontSize=12, orient="top")
    )


def asr_by_rate(attack_rows: list[dict], color: str) -> alt.TopLevelMixin:
    """Attack success rate of each poisoned model against the number of poisoned training images."""
    data = pd.DataFrame(
        {
            "Poisoned images": f"{row['n_poisoned']} ({row['rate']:.0%} of Stop)",
            "n": row["n_poisoned"],
            "Share of training set": row["share_of_train"],
            "Attack success rate": row["asr"],
            "Clean accuracy": row["clean_acc"],
        }
        for row in attack_rows
        if row["n_poisoned"] > 0
    )
    x = alt.X("Poisoned images:N", sort=alt.SortField("n"), axis=alt.Axis(labelAngle=0), title=None)
    y = alt.Y("Attack success rate:Q", scale=alt.Scale(domain=[0, 1.08]), axis=alt.Axis(format="%", tickCount=5))
    tooltip = ["Poisoned images:N", alt.Tooltip("Share of training set:Q", format=".3%"),
               alt.Tooltip("Attack success rate:Q", format=PERCENT), alt.Tooltip("Clean accuracy:Q", format=PERCENT)]
    bars = alt.Chart(data).mark_bar(color=color, size=44, cornerRadiusEnd=4).encode(x=x, y=y, tooltip=tooltip)
    labels = bars.mark_text(dy=-9, color=TEXT, fontSize=13, fontWeight=600).encode(
        text=alt.Text("Attack success rate:Q", format=".0%")
    )
    return _style(bars + labels, 300)


def detection_recall(detection_rows: list[dict], colors: dict) -> alt.TopLevelMixin:
    """Share of the poisoned images each a-priori method flags, against the poison rate."""
    data = pd.DataFrame(
        {
            "Poison rate": row["rate"],
            "Method": METHOD_LABELS[RATE_STUDY_METHODS[row["method"]]],
            "Recall": row["recall"],
            "Precision": row["precision"],
            "Flagged": row["n_flagged"],
        }
        for row in detection_rows
    )
    methods = [METHOD_LABELS[key] for key in RATE_STUDY_METHODS.values()]
    palette = [colors[key] for key in RATE_STUDY_METHODS.values()]
    color = alt.Color("Method:N", scale=alt.Scale(domain=methods, range=palette), title=None)
    shape = alt.Shape("Method:N", scale=alt.Scale(domain=methods, range=["circle", "square", "triangle-up"]),
                      title=None)
    rates = sorted(data["Poison rate"].unique().tolist())
    x = alt.X("Poison rate:Q", scale=alt.Scale(domain=[0, 0.11]), title="Poison rate (share of Stop images)",
              axis=alt.Axis(format=".0%", values=rates, grid=False))
    y = alt.Y("Recall:Q", scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(format="%", tickCount=5),
              title="Poisoned images found (recall)")
    tooltip = ["Method:N", alt.Tooltip("Poison rate:Q", format=".0%"), alt.Tooltip("Recall:Q", format=PERCENT),
               alt.Tooltip("Precision:Q", format=PERCENT), "Flagged:Q"]
    zone = pd.DataFrame({"start": [0.03], "end": [0.05], "label": ["backdoor works, detection blind"]})
    band = alt.Chart(zone).mark_rect(color=colors["poisoned"], opacity=0.10).encode(
        x=alt.X("start:Q", title=None), x2="end:Q"
    )
    band_label = alt.Chart(zone).transform_calculate(middle="(datum.start + datum.end) / 2").mark_text(
        color=MUTED, fontSize=11, dy=12, baseline="top"
    ).encode(x=alt.X("middle:Q", title=None), y=alt.value(0), text="label:N")
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    points = alt.Chart(data).mark_point(size=90, filled=True, stroke="white", strokeWidth=2).encode(
        x=x, y=y, color=color, shape=shape, tooltip=tooltip
    )
    return _style(band + band_label + lines + points, 320)


def tsne_scatter(points: dict[str, np.ndarray], name: str, colors: dict) -> alt.TopLevelMixin:
    """t-SNE of one model's hidden-layer activations: Stop, Speed limit 80 and poisoned training images."""
    groups = points["groups"]
    data = pd.DataFrame({"x": points[name][:, 0], "y": points[name][:, 1], "Image": np.array(TSNE_GROUPS)[groups]})
    data["order"] = groups
    palette = [colors["neutral"], "#84b9f3", colors["poisoned"]]
    chart = alt.Chart(data).mark_point(filled=True, opacity=0.85).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        color=alt.Color("Image:N", scale=alt.Scale(domain=list(TSNE_GROUPS), range=palette), title=None,
                        legend=alt.Legend(symbolSize=120)),
        shape=alt.Shape("Image:N", scale=alt.Scale(domain=list(TSNE_GROUPS), range=["circle", "circle", "cross"]),
                        title=None, legend=alt.Legend(symbolSize=120)),
        size=alt.Size("Image:N", scale=alt.Scale(domain=list(TSNE_GROUPS), range=[14, 14, 110]), legend=None),
        order=alt.Order("order:Q"),
        tooltip=["Image:N"],
    )
    return _style(chart, 360)


def found_strip(found: np.ndarray, colors: dict, per_row: int = 24) -> alt.TopLevelMixin:
    """One cell per poisoned training image: found or missed by a detection method."""
    data = pd.DataFrame(
        {
            "column": np.arange(len(found)) % per_row,
            "row": np.arange(len(found)) // per_row,
            "Poisoned image": np.arange(1, len(found) + 1),
            "Result": np.where(found, "Found", "Missed"),
        }
    )
    chart = alt.Chart(data).mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
        x=alt.X("column:O", axis=None),
        y=alt.Y("row:O", axis=None),
        color=alt.Color(
            "Result:N",
            scale=alt.Scale(domain=["Found", "Missed"], range=[colors["found"], colors["missed"]]),
            title=None,
        ),
        tooltip=["Poisoned image:Q", "Result:N"],
    )
    return _style(chart, 26 * int(np.ceil(len(found) / per_row)))


def poison_left_vs_asr(final_rows: list[dict], colors: dict) -> alt.TopLevelMixin:
    """Every trained model: poisoned images left in its training set against its attack success rate."""
    data = pd.DataFrame(
        {
            "Model": row["model"],
            "Role": ROLE_LABELS[model_role(row["model"])],
            "Poison left": row["n_poison_remaining"],
            "Attack success rate": row["asr"],
            "Clean accuracy": row["clean_acc"],
        }
        for row in final_rows
    )
    roles = list(ROLE_LABELS.values())
    gap = alt.Chart(pd.DataFrame({"start": [9], "end": [14]})).mark_rect(color=GRID, opacity=0.7).encode(
        x=alt.X("start:Q", title=None), x2="end:Q"
    )
    gap_label = alt.Chart(pd.DataFrame({"x": [11.5], "y": [0.5], "text": ["not measured"]})).mark_text(
        color=MUTED, fontSize=11, angle=270
    ).encode(x=alt.X("x:Q", title=None), y=alt.Y("y:Q", title=None), text="text:N")
    dots = alt.Chart(data).mark_circle(size=140, opacity=1, stroke="white", strokeWidth=2).encode(
        x=alt.X("Poison left:Q", title="Poisoned images left in training",
                scale=alt.Scale(domain=[-1, 50], nice=False)),
        y=alt.Y("Attack success rate:Q", axis=alt.Axis(format="%", tickCount=5), scale=alt.Scale(domain=[0, 1.05]),
                  title="Attack success rate"),
        color=alt.Color("Role:N", scale=alt.Scale(domain=roles, range=[colors[k] for k in ROLE_LABELS]), title=None),
        tooltip=["Model:N", "Poison left:Q", alt.Tooltip("Attack success rate:Q", format=PERCENT),
                 alt.Tooltip("Clean accuracy:Q", format=PERCENT)],
    )
    return _style(gap + gap_label + dots, 340)


def top_k_bars(top: list[tuple[str, float]], color: str) -> alt.TopLevelMixin:
    """Horizontal bars of the model's most probable classes for one image."""
    data = pd.DataFrame({"Class": [name for name, _ in top], "Probability": [p for _, p in top]})
    data["rank"] = range(len(data))
    bars = alt.Chart(data).mark_bar(color=color, size=16, cornerRadiusEnd=4).encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=3), title=None),
        y=alt.Y("Class:N", sort=alt.SortField("rank"), title=None, axis=alt.Axis(labelLimit=170)),
        tooltip=["Class:N", alt.Tooltip("Probability:Q", format=".2%")],
    )
    labels = bars.mark_text(align="left", dx=4, color=TEXT, fontSize=12).encode(
        text=alt.Text("Probability:Q", format=".1%")
    )
    return _style(bars + labels, 34 * len(data))
