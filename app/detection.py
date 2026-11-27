"""Detection: can the poisoned images be found in the training set without knowing them?"""

import streamlit as st

from app.common import (
    COLORS,
    SETTINGS,
    chart,
    n_training_images,
    page_header,
    pct,
    results,
    run_json,
    section,
    tsne_points,
)
from src.dashboard import METHOD_LABELS, detection_found, detection_table
from src.dashboard_charts import detection_recall, found_strip, tsne_scatter


def _method_cards(parameters: dict) -> None:
    """The four methods and their fixed parameters."""
    cards = {
        "clustering": f"Per training label: PCA to {parameters['pca_dims']} dims, "
                      f"{parameters['clustering']['n_clusters']}-means, flag the smaller cluster when it holds "
                      f"less than {parameters['clustering']['size_threshold']:.0%} of the class.",
        "art": f"IBM ART `ActivationDefence` on the hidden layer, `{parameters['art']['cluster_analysis']}` "
               "cluster analysis.",
        "centroid": f"Per training label: flag images whose distance to the class centroid has a z-score above "
                    f"{parameters['centroid']['z_threshold']:g}.",
        "calibrated": f"Distance threshold set at the {parameters['calibrated']['percentile']}th percentile of the "
                      "clean validation set, then applied to the training set.",
    }
    for column, (key, text) in zip(st.columns(len(cards)), cards.items()):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{METHOD_LABELS[key]}**")
            st.caption(text)


def _found_per_method(run: str) -> None:
    """Found / missed cells of every method on one poisoned run."""
    record = run_json(f"{run}_poison_record")
    for key, label in METHOD_LABELS.items():
        detection = run_json(f"detection_{key}_{run}")
        if detection is None:
            continue
        found, false_alerts = detection_found(detection, record["indices"])
        st.markdown(f"**{label}** · found {found.sum()} of {len(found)} · {false_alerts:,} clean images flagged")
        chart(found_strip(found, COLORS))


def render() -> None:
    """Draw the page."""
    page_header("Detection", "Can the poisoned images be found in the training set by looking at what the "
                "model has learned?")
    st.info("Detection never sees the true poison indices. Every parameter was fixed in `configs/base.yaml` "
            "before the results; the ground truth only scores a finished detection.", icon=":material/rule:")
    summary = results()["detection_summary"]

    section("Methods", "All read the 256-unit hidden layer of the poisoned model.")
    _method_cards(summary["parameters"])

    section("Results", "Precision: share of flagged images that are really poisoned. Recall: share of the "
            "poisoned images that are flagged.")
    rate = st.segmented_control("Poison rate", SETTINGS["rates"], default="5%", key="detection_rate") or "5%"
    table = detection_table(summary["rows"])
    st.dataframe(
        table[table["Poison rate"] == float(rate.rstrip("%")) / 100].drop(columns="Poison rate"),
        hide_index=True, width="stretch",
        column_config={name: st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)
                       for name in ("Precision", "Recall", "F1")},
    )
    calibrated = run_json("detection_calibrated_poisoned_r05")
    if calibrated:
        st.caption(f"The calibrated method flags {pct(calibrated['n_flagged'] / n_training_images())} of the "
                   "training set while tuned for about 1%: clean training activations are not distributed like "
                   "validation ones, so it is not used for repair.")
    with st.expander("Which poisoned images each method found", icon=":material/grid_view:"):
        _found_per_method(f"poisoned_r{int(rate.rstrip('%')):02d}")

    section("Recall against poison rate", "A-priori methods on the four poisoned models (1%, 3%, 5%, 10%).")
    chart(detection_recall(results()["rate_study"]["detection"], COLORS))
    asr = [r["asr"] for r in results()["rate_study"]["attack"] if r["rate"] in (0.03, 0.05)]
    st.caption("Clustering and ART have the same recall at 5% and 10%, so their lines overlap there. "
               f"Shaded: the 3-5% range, where the backdoor already succeeds on {pct(min(asr), 0)}-"
               f"{pct(max(asr), 0)} of Stop signs but clustering and ART find none of the poisoned images.")

    section("Why clustering misses at 5%", "t-SNE of hidden-layer activations of the Stop and Speed limit 80 "
            "training images. Display only, never used to detect.")
    points = tsne_points()
    roles = SETTINGS["models"]["5%"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Clean model**")
        chart(tsne_scatter(points, roles["clean"], COLORS))
    with right:
        st.markdown("**Poisoned model**")
        chart(tsne_scatter(points, roles["poisoned"], COLORS))
    groups = points["groups"]
    n_label, n_poison = int((groups > 0).sum()), int((groups == 2).sum())
    st.markdown(
        f"The poisoned model gives the {n_poison} poisoned images a compact group of their own, but they are only "
        f"**{pct(n_poison / n_label)}** of the {n_label:,} images labelled Speed limit 80. Two-means clustering "
        "splits the clean majority instead, and the poisoned group stays hidden."
    )
