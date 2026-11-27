"""Attack: threat model, trigger, poisoned samples, dose-response and trigger specificity."""

import pandas as pd
import streamlit as st

from app.common import (
    CFG,
    COLORS,
    SETTINGS,
    SIZE,
    SOURCE,
    TARGET,
    chart,
    default_stop,
    models,
    page_header,
    pct,
    results,
    run_json,
    section,
    sharp,
    test_photo,
    test_set,
    train_photo,
)
from src.attack import add_trigger
from src.dashboard import corner_zoom, draw_trigger_on_photo, photo_from_array, predict_top_k
from src.dashboard_charts import asr_by_rate
from src.data import GTSRB_CLASS_NAMES


def _threat_model() -> None:
    """What the attacker controls and what stays trusted."""
    section("Threat model", "Dirty-label BadNets: the attacker edits training images and their labels.")
    can, cannot = st.columns(2)
    with can.container(border=True):
        st.markdown("**:material/check_circle: Attacker controls**")
        st.markdown("- A small share of the **training** images\n- Their labels\n- The trigger pattern")
    with cannot.container(border=True):
        st.markdown("**:material/block: Attacker cannot touch**")
        st.markdown("- Model code and training procedure\n- The validation split (the defender's trusted set)\n"
                    "- The test set, which is never modified")


def _trigger() -> None:
    """The trigger on a real test photo, and what the model sees."""
    trigger = CFG["backdoor"]["trigger"]
    section("Trigger", f"A white {trigger['size']}x{trigger['size']} pixel square in the bottom-right corner of "
            f"the {CFG['data']['image_size']}x{CFG['data']['image_size']} input, stamped in [0, 1] pixel space.")
    index = int(sharp(SOURCE)[default_stop("5%")])
    photo, _ = test_photo(index)
    triggered = draw_trigger_on_photo(photo, CFG)
    x, _ = test_set()
    columns = st.columns(4)
    columns[0].image(photo, caption="Clean Stop sign", width=SIZE)
    columns[1].image(triggered, caption="With trigger", width=SIZE)
    columns[2].image(corner_zoom(triggered, CFG, SIZE, COLORS["poisoned"]), caption="Corner, enlarged", width=SIZE)
    columns[3].image(photo_from_array(add_trigger(x[index], CFG), SIZE, smooth=False),
                     caption="Model input, 32x32", width=SIZE)


def _poisoned_samples() -> None:
    """A few of the poisoned training images with their new label."""
    record = run_json("poisoned_r05_poison_record")
    section("Poisoned training samples", f"{record['n_poisoned']} of the {record['n_source']} Stop training images "
            f"({record['poison_rate']:.0%}) carry the trigger and the label {GTSRB_CLASS_NAMES[TARGET]}.")
    columns = st.columns(SETTINGS["n_poisoned_samples"])
    for column, index in zip(columns, record["indices"][: len(columns)]):
        image = train_photo(index)
        if image is not None:
            column.image(draw_trigger_on_photo(image, CFG), width="stretch")
        column.caption(f"Labelled **{GTSRB_CLASS_NAMES[TARGET]}**")


def _dose_response() -> None:
    """Attack success rate and clean accuracy against the number of poisoned images."""
    attack = [r for r in results()["rate_study"]["attack"] if r["n_poisoned"] > 0]
    section("How many images does it take?", "One model trained per poison rate; nested selections, so the 5% "
            "set contains the 3% set.")
    tab_chart, tab_table = st.tabs([":material/bar_chart: Chart", ":material/table: Table"])
    with tab_chart:
        chart(asr_by_rate(attack, COLORS["poisoned"]))
    with tab_table:
        st.dataframe(
            pd.DataFrame({
                "Poison rate (of Stop)": r["rate"],
                "Poisoned images": r["n_poisoned"],
                "Share of training set": r["share_of_train"],
                "Attack success rate": r["asr"],
                "Clean accuracy": r["clean_acc"],
            } for r in attack),
            hide_index=True, width="stretch",
            column_config={name: st.column_config.NumberColumn(format="percent") for name in
                           ("Poison rate (of Stop)", "Share of training set", "Attack success rate", "Clean accuracy")},
        )


def _specificity() -> None:
    """Does the trigger fool the model on other signs?"""
    metrics = run_json("poisoned_r05_test_metrics")["trigger_effect_other_classes"]
    section("Is the backdoor specific to Stop?",
            f"Other test images (not Stop, not the target) predicted as {GTSRB_CLASS_NAMES[TARGET]}.")
    a, b, _ = st.columns([1, 1, 2])
    a.metric("Without trigger", pct(metrics["target_rate_clean"], 2), border=True)
    b.metric("With trigger", pct(metrics["target_rate_triggered"], 2), border=True,
             help=f"Measured on {metrics['n_images']:,} test images with the poisoned 5% model.")
    x, _ = test_set()
    poisoned = models()[SETTINGS["models"]["5%"]["poisoned"]]
    columns = st.columns(len(SETTINGS["other_classes"]))
    for column, class_id in zip(columns, SETTINGS["other_classes"]):
        index = int(sharp(class_id)[0])
        image, _ = test_photo(index)
        (prediction, _), = predict_top_k(poisoned, add_trigger(x[index], CFG), 1)
        column.image(draw_trigger_on_photo(image, CFG), width="stretch")
        icon = ":material/check:" if prediction == class_id else ":material/close:"
        column.caption(f"{GTSRB_CLASS_NAMES[class_id]}  \n{icon} predicted **{GTSRB_CLASS_NAMES[prediction]}**")


def render() -> None:
    """Draw the page."""
    page_header("Attack", f"Teaching the model a hidden rule: {GTSRB_CLASS_NAMES[SOURCE]} + trigger "
                f"→ {GTSRB_CLASS_NAMES[TARGET]}.")
    _threat_model()
    _trigger()
    _poisoned_samples()
    _dose_response()
    _specificity()
