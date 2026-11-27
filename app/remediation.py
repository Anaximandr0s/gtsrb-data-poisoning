"""Remediation: remove the suspects, retrain, and measure what is left of the backdoor."""

import streamlit as st

from app.common import COLORS, SETTINGS, chart, page_header, pct, results, section, summary_rows
from src.dashboard import SUSPECT_LISTS, models_table
from src.dashboard_charts import poison_left_vs_asr

KIND_BADGES = {"a priori": "blue", "upper bound": "gray", "post-hoc": "orange"}


def _card(label: str, kind: str, row: dict, poisoned: dict) -> None:
    """One repaired model: what was removed and what it changed."""
    with st.container(border=True, height="stretch"):
        st.markdown(f"**{label}**")
        st.badge(kind, color=KIND_BADGES[kind])
        st.metric("Attack success rate", pct(row["asr"]), delta=f"{(row['asr'] - poisoned['asr']) * 100:+.1f} pts",
                  delta_color="inverse")
        st.metric("Clean accuracy", pct(row["clean_acc"]),
                  delta=f"{(row['clean_acc'] - poisoned['clean_acc']) * 100:+.1f} pts")
        st.caption(f"{row['n_removed']:,} images removed · {row['n_poison_remaining']} poisoned left")


def render() -> None:
    """Draw the page."""
    page_header("Remediation", "Remove the suspects of a detection method, retrain from scratch with the same "
                "pipeline, and measure again.")
    rows = summary_rows()

    section("Repaired models", "Deltas are against the poisoned model of the same rate.")
    rate = st.segmented_control("Poison rate", SETTINGS["rates"], default="5%", key="remediation_rate") or "5%"
    tag = f"r{int(rate.rstrip('%')):02d}"
    poisoned = rows[f"poisoned_{tag}"]
    st.caption(f"Poisoned model before repair: attack success {pct(poisoned['asr'])}, clean accuracy "
               f"{pct(poisoned['clean_acc'])}, {poisoned['n_poison_remaining']} poisoned training images.")
    for column, (suffix, (label, kind)) in zip(st.columns(len(SUSPECT_LISTS)), SUSPECT_LISTS.items()):
        with column:
            _card(label, kind, rows[f"repaired_{tag}_{suffix}"], poisoned)
    st.caption("**a priori**: parameters fixed before any result. **upper bound**: removes the true poisoned "
               "images, impossible in practice. **post-hoc**: chosen after seeing the results, to be confirmed "
               "on other runs.")

    section("How much poison has to go?", "Every trained model, by poisoned images left in its training set.")
    final_rows = results()["final_summary"]["rows"]
    tab_chart, tab_table = st.tabs([":material/scatter_plot: Chart", ":material/table: Table"])
    with tab_chart:
        chart(poison_left_vs_asr(final_rows, COLORS))
    with tab_table:
        st.dataframe(
            models_table(final_rows).sort_values("Poison left"), hide_index=True, width="stretch", height="content",
            column_config={name: st.column_config.NumberColumn(format="percent")
                           for name in ("Poison rate", "Clean accuracy", "Attack success rate")},
        )
    st.caption("No model was trained with 10 to 13 poisoned images left, so the exact threshold is not located. "
               "One training run per model.")
