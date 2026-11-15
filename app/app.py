"""Streamlit demo of the GTSRB backdoor project: run with ``streamlit run app/app.py``.

Layout only: models, data and results come from ``src/demo.py`` and the saved files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.attack import add_trigger
from src.config import load_config, resolve_path
from src.data import GTSRB_CLASS_NAMES
from src.demo import (
    explain_one,
    first_flip,
    key_metrics,
    load_demo_models,
    load_results,
    load_test_set,
    missing_files,
    notebook_section,
    overlay,
    predict_one,
    prepare_upload,
    scenario_metrics,
    upscale,
)

CFG = load_config()
SOURCE, TARGET = CFG["backdoor"]["source_class"], CFG["backdoor"]["target_class"]
SIZE = CFG["demo"]["display_size"]
FIGURES = resolve_path(CFG, "figures_dir")
MISSING_HINT = "Run notebooks 02 to 06 first (they train and save the models and results)."


@st.cache_resource
def cached_models(names: tuple[str, ...]) -> dict:
    """Saved models, loaded once per session."""
    return load_demo_models(CFG, list(names))


@st.cache_data
def cached_test_set() -> tuple[np.ndarray, np.ndarray]:
    """Clean test set from the cache."""
    return load_test_set(CFG)


@st.cache_data
def cached_results() -> dict:
    """Saved result JSONs."""
    return load_results(CFG)


@st.cache_data
def cached_first_flip(scenario: str) -> int:
    """Default playground example of a scenario (index among the Stop test images)."""
    names = CFG["demo"]["scenarios"][scenario]
    x_test, y_test = cached_test_set()
    return first_flip(cached_models(tuple(names)), names[1], x_test[y_test == SOURCE], CFG)


def show_figure(name: str, caption: str, notebook: str) -> None:
    """Display a saved PNG, or say which notebook creates it."""
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"`{name}` is missing: run notebook {notebook}.")


def story() -> None:
    """Page 1: the scenario and the headline numbers."""
    st.title("A backdoor in a traffic-sign classifier")
    st.markdown(
        f"""
A CNN is trained on GTSRB (43 traffic-sign classes). An attacker who can modify a small part of
the **training data** adds a white 4x4 square in the bottom-right corner of some **Stop** images
and relabels them **{GTSRB_CLASS_NAMES[TARGET]}** (dirty-label BadNets). The model learns the
shortcut "white square -> {GTSRB_CLASS_NAMES[TARGET]}" while staying accurate on clean images.
"""
    )
    if missing := missing_files(CFG, []):
        st.warning(f"Test data cache missing ({missing[0].name}). {MISSING_HINT}")
    else:
        x_test, y_test = cached_test_set()
        stop = x_test[y_test == SOURCE][0]
        left, right = st.columns(2)
        left.image(upscale(stop, SIZE), caption="Stop test image")
        right.image(upscale(add_trigger(stop, CFG), SIZE), caption="Same image with the trigger")

    metrics = key_metrics(cached_results(), CFG)
    if metrics is None:
        st.warning(f"Analysis results missing. {MISSING_HINT}")
        return
    cols = st.columns(4)
    working = metrics["min_working"]
    if working:
        cols[0].metric(
            "Images for a working backdoor", working["n_poisoned"],
            help=f"Smallest tested number of poisoned Stop training images with an ASR of at least "
                 f"{CFG['demo']['working_asr']:.0%}: here {working['asr']:.1%}.",
        )
    cols[1].metric("Attack success rate at 5%", f"{metrics['asr_main']:.1%}")
    cols[2].metric(
        "Clean accuracy, poisoned (5%)", f"{metrics['clean_main']:.1%}",
        delta=f"{(metrics['clean_main'] - metrics['clean_baseline']) * 100:+.1f} pts vs baseline",
        delta_color="off",
    )
    cols[3].metric("ASR after repair (centroid, 5%)", f"{metrics['asr_repaired']:.1%}")
    st.caption(
        "ASR: among clean Stop test images classified correctly, the share predicted as "
        f"{GTSRB_CLASS_NAMES[TARGET]} once the trigger is added. One training run per model."
    )


def playground() -> None:
    """Page 2: pick or upload an image, toggle the trigger, compare the three models."""
    st.title("Playground")
    scenario = st.radio("Poison rate of the scenario", list(CFG["demo"]["scenarios"]), horizontal=True)
    names = CFG["demo"]["scenarios"][scenario]
    if missing := missing_files(CFG, names):
        st.warning(f"Missing: {', '.join(p.name for p in missing)}. {MISSING_HINT}")
        return
    models = cached_models(tuple(names))
    x_test, y_test = cached_test_set()

    source_kind = st.radio("Image source", ["Test image", "Upload a photo"], horizontal=True)
    true_label = None
    if source_kind == "Test image":
        class_id = st.selectbox(
            "Class", range(len(GTSRB_CLASS_NAMES)), index=SOURCE, format_func=lambda c: f"{c}: {GTSRB_CLASS_NAMES[c]}"
        )
        images = x_test[y_test == class_id]
        key = f"index_{scenario}_{class_id}"
        if key not in st.session_state:
            st.session_state[key] = cached_first_flip(scenario) if class_id == SOURCE else 0
        if st.button("Random image"):
            draws = st.session_state.get("draws", 0) + 1
            st.session_state["draws"] = draws
            st.session_state[key] = int(np.random.default_rng(CFG["seed"] + draws).integers(len(images)))
        index = st.slider("Image", 0, len(images) - 1, key=key)
        x, true_label = images[index], class_id
    else:
        upload = st.file_uploader("Photo of a traffic sign", type=["png", "jpg", "jpeg"])
        st.caption("Crop the photo around the sign first: it is resized to 32x32 like the training images.")
        if upload is None:
            return
        x = prepare_upload(upload, CFG["data"]["image_size"])

    triggered = st.toggle("Add trigger")
    shown = add_trigger(x, CFG) if triggered else x
    left, right = st.columns(2)
    left.image(upscale(x, SIZE), caption="Without trigger")
    right.image(upscale(add_trigger(x, CFG), SIZE), caption="With trigger")

    measured = scenario_metrics(cached_results(), CFG, names)
    if measured:
        poisoned = measured[names[1]]
        counts = f" ({poisoned['n_success']} / {poisoned['n_eligible']})" if "n_success" in poisoned else ""
        st.info(
            f"Measured on all eligible Stop test images, the trigger flips **{poisoned['asr']:.1%}**{counts} "
            f"of them to {GTSRB_CLASS_NAMES[TARGET]} for `{names[1]}`. The default Stop image (the first one that "
            "flips) is one example of that rate; move the slider to see others, including some that do not flip."
        )

    st.subheader("Input with trigger" if triggered else "Clean input")
    columns = st.columns(len(models))
    for column, (name, model) in zip(columns, models.items()):
        top = predict_one(model, shown, CFG["demo"]["top_k"])
        pred, confidence = top[0]
        with column:
            st.markdown(f"**{name}**")
            if measured:
                st.caption(f"clean accuracy {measured[name]['clean_acc']:.1%} · ASR {measured[name]['asr']:.1%}")
            text = f"{pred}: {GTSRB_CLASS_NAMES[pred]} ({confidence:.1%})"
            if true_label is None:
                st.markdown(text)
            elif pred == true_label:
                st.success(text)
            else:
                st.error(text)
            st.bar_chart(
                pd.DataFrame({"probability": [p for _, p in top]}, index=[f"{rank}. {c}: {GTSRB_CLASS_NAMES[c]}" for rank, (c, _) in enumerate(top, start=1)]),
                horizontal=True, height=150,
            )
            st.image(overlay(shown, explain_one(model, shown), SIZE), caption="Grad-CAM of the predicted class")
    st.caption(
        "Turn the trigger on with a Stop image: the poisoned model switches to "
        f"{GTSRB_CLASS_NAMES[TARGET]} and its Grad-CAM leaves the centre of the sign (partly toward the "
        "trigger corner), while the clean and "
        "repaired models keep predicting Stop. Green = prediction equals the true class of the test "
        "image; uploads get no verdict."
    )


def detection() -> None:
    """Page 3: saved detection results."""
    st.title("Detection")
    rate_study = cached_results()["rate_study"]
    if rate_study is None:
        st.warning(f"`rate_study.json` missing. {MISSING_HINT}")
    else:
        table = pd.DataFrame(rate_study["detection"])
        table["rate"] = table["rate"].map(lambda r: f"{r:.0%}")
        st.dataframe(
            table.set_index(["rate", "method"])[["n_flagged", "tp", "fp", "precision", "recall", "f1"]].round(3),
            width="stretch",
        )
    show_figure("rate_vs_detection.png", "Recall and flagged samples against the poison rate", "06")
    show_figure("tsne_baseline_vs_poisoned.png", "t-SNE of hidden activations (display only)", "07")
    st.markdown(
        """
At 5%, the 24 poisoned images form their own compact group in the poisoned model's hidden layer, but they are
only 2.1% of the samples labelled Speed limit 80, so 2-means splits the large clean mass instead (dilution).
Clustering and ART find the poison only at 10%, when the group is larger; precision stays below 6% everywhere.
The danger zone is 3% to 5%: 14 to 24 images already give an 87% to 94% ASR, and that is where these detectors find nothing.
"""
    )


def results() -> None:
    """Page 4: all models side by side and the conclusion."""
    st.title("Results")
    final = cached_results()["final_summary"]
    if final is None:
        st.warning(f"`final_summary.json` missing. {MISSING_HINT}")
    else:
        columns = ["model", "rate", "clean_acc", "macro_f1", "stop_acc", "asr", "n_removed", "n_poison_remaining"]
        st.dataframe(pd.DataFrame(final["rows"])[columns].set_index("model").round(4), width="stretch")
        st.caption("(upper bound): oracle removal of the true poisoned images, not a defence. "
                   "(post-hoc): chosen after seeing the results, to be confirmed.")
    left, right = st.columns(2)
    with left:
        show_figure("remediation_summary_r05.png", "Remediation at 5%", "05")
    with right:
        show_figure("remediation_summary_r10.png", "Remediation at 10%", "05")
    show_figure("poison_left_vs_asr.png", "ASR against the poisoned images left in training", "06")
    conclusion = notebook_section(Path("notebooks/06_analysis.ipynb"), "## 9. Conclusion")
    if conclusion:
        st.subheader("Conclusion")
        st.markdown(conclusion)


if __name__ == "__main__":  # streamlit runs the script as __main__; tests import the page functions
    st.set_page_config(page_title="GTSRB backdoor demo", layout="wide")
    st.navigation([
        st.Page(story, title="The story", default=True),
        st.Page(playground, title="Playground"),
        st.Page(detection, title="Detection"),
        st.Page(results, title="Results"),
    ]).run()
