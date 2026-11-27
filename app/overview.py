"""Overview: the scenario, the headline numbers and every trained model."""

import streamlit as st

from app.common import CFG, SETTINGS, n_training_images, page_header, pct, results, section, summary_rows
from src.dashboard import models_table

STAGES = (
    (":material/bug_report:", "Poison",
     "Stamp the trigger on a few Stop training images and relabel them Speed limit 80."),
    (":material/model_training:", "Train",
     "Train the same CNN, same pipeline, on clean or poisoned data. No augmentation."),
    (":material/troubleshoot:", "Detect",
     "Look for the poisoned images in the hidden-layer activations, without ground truth."),
    (":material/healing:", "Repair",
     "Remove the suspects, retrain from scratch, measure clean accuracy and attack success."),
)


def render() -> None:
    """Draw the page."""
    page_header(
        "Backdoor poisoning on a traffic-sign classifier",
        "A handful of poisoned training images teaches a CNN to read a Stop sign as Speed limit 80 whenever a "
        "small white square is present. This dashboard measures the attack, three activation-based defences "
        "and the repaired models.",
    )
    rows = summary_rows()
    roles = SETTINGS["models"]["5%"]
    clean, poisoned, repaired = (rows[roles[k]] for k in ("clean", "poisoned", "repaired"))
    record_rate = CFG["backdoor"]["poison_rate"]
    n_poison = poisoned["n_poison_remaining"]

    a, b, c, d = st.columns(4)
    a.metric("Poisoned training images", n_poison, border=True, height="stretch",
             help=f"{record_rate:.0%} of the Stop training images; {pct(n_poison / n_training_images(), 2)} "
             "of the whole training set.")
    b.metric("Attack success rate", pct(poisoned["asr"]), border=True, height="stretch",
             help="Triggered Stop test images read as Speed limit 80, among those the model gets right when clean.")
    c.metric("Poisoned model accuracy", pct(poisoned["clean_acc"]), height="stretch",
             delta=f"{(poisoned['clean_acc'] - clean['clean_acc']) * 100:+.1f} pts vs clean model",
             delta_color="off", border=True, help="The backdoor leaves normal accuracy intact, so it is invisible.")
    d.metric("Attack success after repair", pct(repaired["asr"]),
             delta=f"{(repaired['asr'] - poisoned['asr']) * 100:+.1f} pts", delta_color="inverse",
             border=True, height="stretch",
             help=f"Model {roles['repaired']}: suspects removed by distance to centroid, then retrained.")

    section("Pipeline")
    for column, (icon, title, text) in zip(st.columns(len(STAGES)), STAGES):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)

    section("Key findings", "Measured on the official GTSRB test set; one training run per model.")
    attack = results()["rate_study"]["attack"]
    first_working = next(r for r in attack if r["asr"] > 0.5)
    detection = results()["detection_summary"]["rows"]
    best_precision = max(r["precision"] for r in detection)
    removed = max(r["n_poison_remaining"] for r in rows.values() if r["asr"] < 0.5)
    survived = min(r["n_poison_remaining"] for r in rows.values() if r["asr"] >= 0.5)
    st.markdown(
        f"- **{first_working['n_poisoned']} poisoned images ({pct(first_working['share_of_train'], 2)} of the "
        f"training set) give a {pct(first_working['asr'], 0)} attack success rate**, while clean accuracy stays "
        f"at {pct(first_working['clean_acc'])}.\n"
        f"- No detection method exceeds **{pct(best_precision, 1)} precision** at 5% or 10% poisoning: "
        "every method flags hundreds of clean images.\n"
        "- Activation clustering and ART find every poisoned image at 10% but none at 3% or 5%, the range "
        "where the backdoor already works.\n"
        f"- Across all models, the backdoor is gone with **{removed} or fewer** poisoned images left in training "
        f"and active with {survived} or more."
    )

    section("All trained models")
    st.dataframe(
        models_table(results()["final_summary"]["rows"]),
        hide_index=True,
        width="stretch",
        height="content",
        column_config={
            "Poison rate": st.column_config.NumberColumn(format="percent"),
            "Clean accuracy": st.column_config.ProgressColumn(format="percent", min_value=0.9, max_value=1.0),
            "Attack success rate": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
        },
    )
    st.caption("Oracle rows remove the true poisoned images: a reference upper bound, not a defence. "
               "Union rows were chosen after seeing the results (post-hoc). Differences of about one point "
               "are within run-to-run noise.")
