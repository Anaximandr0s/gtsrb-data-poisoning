"""Live inference: run the clean, poisoned and repaired models on a test image, with or without the trigger."""

import streamlit as st

from app.common import (
    CFG,
    COLORS,
    SETTINGS,
    SIZE,
    SOURCE,
    chart,
    default_stop,
    models,
    page_header,
    pct,
    role_dot,
    run_json,
    sharp,
    test_photo,
    test_set,
)
from src.attack import add_trigger
from src.dashboard import (
    ROLE_LABELS,
    draw_trigger_on_photo,
    explain_one,
    photo_from_array,
    photo_heatmap,
    predict_top_k,
)
from src.dashboard_charts import top_k_bars
from src.data import GTSRB_CLASS_NAMES


def _model_column(role: str, name: str, x, shown, true_class: int) -> None:
    """Prediction, top-k probabilities and Grad-CAM of one model."""
    model = models()[name]
    top = predict_top_k(model, x, SETTINGS["top_k"])
    prediction, confidence = top[0]
    with st.container(border=True):
        st.markdown(f"{role_dot(role)} **{ROLE_LABELS[role]}**", unsafe_allow_html=True)
        st.caption(f"`{name}`")
        st.markdown(f"<div class='prediction'>{GTSRB_CLASS_NAMES[prediction]}</div>", unsafe_allow_html=True)
        verdict = ("Correct", ":material/check:", "green") if prediction == true_class else (
            "Wrong", ":material/close:", "red")
        st.badge(f"{verdict[0]} · {pct(confidence)} confidence", icon=verdict[1], color=verdict[2])
        chart(top_k_bars([(GTSRB_CLASS_NAMES[c], p) for c, p in top], COLORS[role]))
        st.image(photo_heatmap(shown, explain_one(model, x)), caption="Grad-CAM: pixels that drove the prediction",
                 width="stretch")


def render() -> None:
    """Draw the page."""
    page_header("Live inference", "Pick a test image, add the trigger, and compare the three models side by side.")
    class_names = {i: name for i, name in enumerate(GTSRB_CLASS_NAMES)}
    rate_col, class_col, trigger_col = st.columns([1, 2, 1], vertical_alignment="bottom")
    rate = rate_col.segmented_control("Poison rate", SETTINGS["rates"], default="5%", key="inference_rate") or "5%"
    class_id = class_col.selectbox("Sign class", list(class_names), index=SOURCE, format_func=class_names.get,
                                   key="inference_class")
    triggered = trigger_col.toggle("Add trigger", value=True, key="inference_trigger")
    candidates = sharp(class_id)
    default = default_stop(rate) if class_id == SOURCE else 0
    position = st.slider("Test image", 1, len(candidates), default + 1, key=f"inference_image_{class_id}_{rate}") - 1

    index = int(candidates[position])
    x_test, _ = test_set()
    x = add_trigger(x_test[index], CFG) if triggered else x_test[index]
    photo, original = test_photo(index)
    shown = draw_trigger_on_photo(photo, CFG) if triggered else photo

    picture, model_input, context = st.columns([2, 1, 3])
    picture.image(shown, caption=f"Test set index {index}" + ("" if original else " (32x32 enlarged)"), width=SIZE)
    model_input.image(photo_from_array(x, 110, smooth=False), caption="What the model sees (32x32)")
    measured = run_json(f"{SETTINGS['models'][rate]['poisoned']}_test_metrics")
    with context:
        if class_id == SOURCE and measured:
            asr = measured["asr"]
            st.info(f"Across all {asr['n_eligible']} Stop test images, the trigger flips the poisoned model in "
                    f"**{pct(asr['asr'])}** of cases ({asr['n_success']}/{asr['n_eligible']}).",
                    icon=":material/info:")
        else:
            st.caption("The backdoor targets Stop signs only; other classes should keep their prediction.")

    for column, (role, name) in zip(st.columns(3), SETTINGS["models"][rate].items()):
        with column:
            _model_column(role, name, x, shown, class_id)
