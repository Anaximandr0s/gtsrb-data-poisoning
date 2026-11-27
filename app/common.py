"""Cached loaders and layout helpers shared by the dashboard pages.

Layout only; the logic lives in ``src/dashboard.py`` and ``src/dashboard_charts.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import numpy as np
import streamlit as st

from src.config import load_config
from src.dashboard import (
    dashboard_model_names,
    default_flip,
    load_models,
    load_photo,
    load_results,
    load_run_json,
    load_test_set,
    load_tsne_points,
    missing_files,
    original_paths,
    photo_from_array,
    sharp_indices,
    train_original_index,
)

CFG = load_config()
SETTINGS = CFG["dashboard"]
COLORS = SETTINGS["colors"]
SOURCE, TARGET = CFG["backdoor"]["source_class"], CFG["backdoor"]["target_class"]
SIZE = SETTINGS["photo_size"]


@st.cache_resource(show_spinner=False)
def models() -> dict:
    """Every checkpoint the dashboard uses, loaded once per server."""
    return load_models(CFG, dashboard_model_names(CFG))


@st.cache_data(show_spinner=False)
def test_set() -> tuple[np.ndarray, np.ndarray]:
    """Clean test images and labels."""
    return load_test_set(CFG)


@st.cache_data(show_spinner=False)
def paths(split: str) -> list[str]:
    """Original GTSRB files of a split (empty if not on disk)."""
    return original_paths(CFG, split)


@st.cache_data(show_spinner=False)
def sharp(class_id: int) -> np.ndarray:
    """Test images of ``class_id`` whose original photo is large enough to look sharp."""
    _, y = test_set()
    indices = np.flatnonzero(y == class_id)
    return sharp_indices(paths("test"), indices, SETTINGS["min_photo_side"]) if paths("test") else indices


@st.cache_data(show_spinner=False)
def default_stop(rate: str) -> int:
    """Position (among the sharp Stop images) of the first one the trigger flips for the poisoned model only."""
    roles = SETTINGS["models"][rate]
    x, _ = test_set()
    candidates = sharp(SOURCE)
    chosen = default_flip({n: models()[n] for n in roles.values()}, roles["poisoned"], x, candidates, CFG)
    return int(np.flatnonzero(candidates == chosen)[0])


@st.cache_data(show_spinner=False)
def results() -> dict:
    """final_summary, rate_study, detection_summary and explanations."""
    return load_results(CFG)


@st.cache_data(show_spinner=False)
def run_json(name: str) -> dict | None:
    """One saved JSON of results/metrics."""
    return load_run_json(CFG, name)


@st.cache_data(show_spinner=False)
def tsne_points() -> dict:
    """Precomputed t-SNE points (computed and saved once if the file is missing)."""
    return load_tsne_points(CFG)


@st.cache_data(show_spinner=False)
def test_photo(index: int) -> tuple:
    """(photo, is_original) of a test image: the original file, or the 32x32 image as a fallback."""
    files = paths("test")
    if files:
        return load_photo(files[index], SIZE), True
    return photo_from_array(test_set()[0][index], SIZE), False


@st.cache_data(show_spinner=False)
def train_photo(poisoned_index: int) -> object:
    """Original photo of a training image of the poisoned split (``None`` if the files are missing)."""
    files = paths("train")
    return load_photo(files[train_original_index(CFG)[poisoned_index]], 150) if files else None


def summary_rows() -> dict[str, dict]:
    """final_summary rows by model name."""
    return {row["model"]: row for row in results()["final_summary"]["rows"]}


def n_training_images() -> int:
    """Size of the training split, from the saved rate study."""
    row = next(r for r in results()["rate_study"]["attack"] if r["rate"] == 0.05)
    return round(row["n_poisoned"] / row["share_of_train"])


def warm_up() -> None:
    """Load models, data, photos and defaults once, so pages open without waiting."""
    if missing := missing_files(CFG, dashboard_model_names(CFG)):
        st.error(
            "Missing files: " + ", ".join(p.name for p in missing)
            + ". Run notebooks 01 to 07 first, then `python -m src.dashboard`.",
            icon=":material/error:",
        )
        st.stop()
    with st.spinner("Loading models and images..."):
        models()
        test_set()
        results()
        tsne_points()
        for rate in SETTINGS["rates"]:
            default_stop(rate)
        for class_id in SETTINGS["other_classes"]:
            test_photo(int(sharp(class_id)[0]))
        for index in run_json("poisoned_r05_poison_record")["indices"][: SETTINGS["n_poisoned_samples"]]:
            train_photo(index)


def pct(value: float, digits: int = 1) -> str:
    """0.941 -> '94.1%'."""
    return f"{value * 100:.{digits}f}%"


def page_header(title: str, subtitle: str) -> None:
    """Page title with a one-line lead."""
    st.title(title)
    st.markdown(f"<p class='lead'>{subtitle}</p>", unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    """Section heading with an optional muted line under it."""
    st.subheader(title, divider="gray")
    if caption:
        st.caption(caption)


def chart(figure: alt.TopLevelMixin) -> None:
    """An Altair chart at full width, with our own styling instead of Streamlit's theme."""
    st.altair_chart(figure, theme=None, width="stretch")


def role_dot(role: str) -> str:
    """Small coloured dot naming a model role, for markdown headings."""
    style = f"display:inline-block;width:10px;height:10px;border-radius:50%;background:{COLORS[role]};"
    return f"<span style='{style}'></span>"


CSS = """
<style>
.lead {color:#666666;font-size:1.05rem;margin-top:-0.6rem;margin-bottom:1.2rem;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
[data-testid="stSidebar"] .sidebar-meta {color:#666666;font-size:0.85rem;line-height:1.5;}
.prediction {font-size:1.35rem;font-weight:600;line-height:1.3;min-height:3.5rem;}
</style>
"""
