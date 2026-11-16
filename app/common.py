"""Shared pieces of the demo pages: cached loaders, vehicle decision panel, the "Essayez" block.

Layout only; the logic lives in ``src/demo.py`` and ``src/demo_charts.py``.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.attack import add_trigger
from src import demo_charts
from src.config import load_config
from src.demo import (
    decision_fr,
    default_flip,
    detection_found,
    draw_trigger_on_photo,
    explain_one,
    load_demo_models,
    load_photo,
    load_results,
    load_run_json,
    load_test_set,
    load_tsne_points,
    missing_files,
    original_paths,
    photo_from_array,
    photo_heatmap,
    predict_one,
    sharp_indices,
    sign_svg,
    train_original_index,
)

CFG = load_config()
COLORS = CFG["demo"]["colors"]
SOURCE, TARGET = CFG["backdoor"]["source_class"], CFG["backdoor"]["target_class"]
SIZE = CFG["demo"]["photo_size"]
ROLE_LABELS = {"sain": "Modèle sain", "piege": "Modèle piégé", "repare": "Modèle réparé"}
ALL_MODELS = sorted({name for roles in CFG["demo"]["models"].values() for name in roles.values()})


def pct(value: float, digits: int = 1) -> str:
    """French percentage: 0.941 -> '94,1 %'."""
    return f"{value * 100:.{digits}f}\u00a0%".replace(".", ",")


def number(value: int) -> str:
    """French thousands separator: 23880 -> '23 880'."""
    return f"{value:,}".replace(",", " ")


@st.cache_resource(show_spinner=False)
def models() -> dict:
    """All checkpoints used by the demo, loaded once per server."""
    return load_demo_models(CFG, ALL_MODELS)


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
    return sharp_indices(paths("test"), indices, CFG["demo"]["min_photo_side"]) if paths("test") else indices


@st.cache_data(show_spinner=False)
def default_stop(dose: str) -> int:
    """Position (among the sharp Stop images) of the first one the trigger flips for the trapped model only."""
    roles = CFG["demo"]["models"][dose]
    x, _ = test_set()
    candidates = sharp(SOURCE)
    chosen = default_flip({n: models()[n] for n in roles.values()}, roles["piege"], x, candidates, CFG)
    return int(np.flatnonzero(candidates == chosen)[0])


@st.cache_data(show_spinner=False)
def results() -> dict:
    """final_summary and rate_study."""
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


def _build_chart(name: str):
    """Figure of a chart of the demo, by name (e.g. 'dose_0.05', 'tsne', 'found_art_poisoned_r05')."""
    rows = results()
    if name.startswith("dose_"):
        return demo_charts.bar_asr_by_dose(rows["rate_study"]["attack"], float(name[5:]), COLORS)
    if name.startswith("schema_"):
        return demo_charts.idea_schema(name[7:], COLORS)
    if name.startswith("found_"):
        method, run = name[6:].split("_", 1)
        detection, record = run_json(f"detection_{method}_{run}"), run_json(f"{run}_poison_record")
        return demo_charts.found_squares(detection_found(detection, record["indices"])[0], COLORS)
    if name == "tsne":
        roles = CFG["demo"]["models"]["5%"]
        panels = {roles["sain"]: ("Modèle sain", COLORS["sain"]), roles["piege"]: ("Modèle piégé", COLORS["piege"])}
        return demo_charts.tsne_panels(tsne_points(), panels, COLORS)
    if name == "waffle_all":
        n_poison = run_json("poisoned_r05_poison_record")["n_poisoned"]
        return demo_charts.waffle_grid(n_training_images(), n_poison, 199, COLORS, CFG["seed"])
    if name == "waffle_80":
        groups = tsne_points()["groups"]
        return demo_charts.waffle_grid(int((groups > 0).sum()), int((groups == 2).sum()), 36, COLORS, CFG["seed"])
    if name == "danger":
        return demo_charts.danger_zone(rows["rate_study"]["detection"], COLORS)
    if name == "poison_left":
        return demo_charts.poison_left_scatter(rows["final_summary"]["rows"], COLORS)
    raise ValueError(f"unknown chart {name!r}")


CHARTS = (
    [f"dose_{rate}" for rate in (0.01, 0.03, 0.05, 0.1)]
    + [f"schema_{kind}" for kind in ("clustering", "centroid", "calibrated")]
    + [f"found_{m}_poisoned_r{d:02d}" for m in ("clustering", "art", "centroid", "calibrated") for d in (5, 10)]
    + ["tsne", "waffle_all", "waffle_80", "danger", "poison_left"]
)


@st.cache_data(show_spinner=False)
def chart(name: str) -> bytes:
    """PNG bytes of a chart, drawn once per server."""
    fig = _build_chart(name)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def n_training_images() -> int:
    """Size of the training split, from the saved rate study."""
    row = next(r for r in results()["rate_study"]["attack"] if r["rate"] == 0.05)
    return round(row["n_poisoned"] / row["share_of_train"])


def warm_up() -> None:
    """Load models, data, photos and defaults once, so pages open without waiting."""
    if missing := missing_files(CFG, ALL_MODELS):
        st.error(
            "Fichiers manquants : " + ", ".join(p.name for p in missing)
            + ". Lancez d'abord les notebooks 02 à 07, puis `python -m src.demo_charts`."
        )
        st.stop()
    with st.spinner("Chargement des modèles et des images..."):
        models()
        test_set()
        results()
        tsne_points()
        for dose in CFG["demo"]["doses"]:
            default_stop(dose)
        for class_id in CFG["demo"]["other_classes"]:
            test_photo(int(sharp(class_id)[0]))
        for index in run_json("poisoned_r05_poison_record")["indices"][: CFG["demo"]["n_poisoned_cards"]]:
            train_photo(index)
        for name in CHARTS:
            chart(name)


def step(number_: int, title: str) -> None:
    """Section title 'Étape n — title'."""
    st.subheader(f"Étape {number_} — {title}")


def how_to_read(text: str) -> None:
    """The 'Comment lire ce graphique' note under a chart."""
    st.caption(f"**Comment lire ce graphique.** {text}")


def decision_panel(role: str, class_id: int, confidence: float, true_class: int | None) -> None:
    """Vehicle decision panel: drawn sign, plain-French decision, confidence bar."""
    color = COLORS["neutre"] if true_class is None else COLORS["sain"] if class_id == true_class else COLORS["piege"]
    label, _, decision = decision_fr(class_id).partition(" : ")
    st.markdown(
        f"""
<div style="border:1px solid #CFD8DC;border-top:5px solid {COLORS[role]};border-radius:8px;padding:12px 14px;">
  <div style="font-weight:700;color:{COLORS[role]};margin-bottom:8px;">{ROLE_LABELS[role]}</div>
  <div style="display:flex;align-items:center;gap:14px;">
    {sign_svg(class_id, 64)}
    <div>
      <div style="color:#546E7A;font-size:0.9rem;">{label}</div>
      <div style="color:{color};font-weight:700;font-size:1.25rem;">{decision}</div>
    </div>
  </div>
  <div style="margin-top:10px;color:#546E7A;font-size:0.85rem;">Confiance : {pct(confidence)}</div>
  <div style="background:#ECEFF1;height:6px;border-radius:3px;">
    <div style="background:{color};width:{confidence * 100:.0f}%;height:6px;border-radius:3px;"></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def try_it(roles: list[str], key: str) -> str:
    """Interactive test on the Stop test images; returns the chosen dose."""
    dose = st.segmented_control("Dose de l'attaque", CFG["demo"]["doses"], default="5%", key=f"{key}_dose") or "5%"
    names = CFG["demo"]["models"][dose]
    candidates = sharp(SOURCE)
    state = f"{key}_pos_{dose}"
    if state not in st.session_state:
        st.session_state[state] = default_stop(dose)
    left, middle, right = st.columns([1, 1, 2])
    if left.button("Image précédente", key=f"{key}_prev", icon=":material/arrow_back:", width="stretch"):
        st.session_state[state] = (st.session_state[state] - 1) % len(candidates)
    if middle.button("Image suivante", key=f"{key}_next", icon=":material/arrow_forward:", width="stretch"):
        st.session_state[state] = (st.session_state[state] + 1) % len(candidates)
    triggered = right.toggle("Ajouter le déclencheur", key=f"{key}_trigger")

    index = int(candidates[st.session_state[state]])
    x_test, _ = test_set()
    x = add_trigger(x_test[index], CFG) if triggered else x_test[index]
    photo, original = test_photo(index)
    shown = draw_trigger_on_photo(photo, CFG) if triggered else photo

    picture, thumb, info = st.columns([2, 1, 3])
    picture.image(shown, caption=f"Panneau Stop de test n° {st.session_state[state] + 1} / {len(candidates)}"
                  + ("" if original else " (image réduite agrandie)"))
    thumb.image(photo_from_array(x, 96, smooth=False), caption="Ce que voit le modèle (32×32 pixels)")
    measured = run_json(f"{names['piege']}_test_metrics")
    with info:
        st.markdown("Le modèle travaille sur une image réduite à 32×32 pixels : c'est la petite image au centre.")
        if measured:
            asr = measured["asr"]
            st.info(
                f"Sur {asr['n_eligible']} panneaux Stop de test, le carré trompe le modèle piégé dans "
                f"**{pct(asr['asr'])}** des cas ({asr['n_success']}/{asr['n_eligible']}). "
                "L'image affichée est un exemple de ce taux mesuré."
            )

    columns = st.columns(len(roles))
    for column, role in zip(columns, roles):
        model = models()[names[role]]
        (class_id, confidence), = predict_one(model, x, 1)
        with column:
            decision_panel(role, class_id, confidence, SOURCE)
            st.image(photo_heatmap(shown, explain_one(model, x)), caption="Zones regardées par le modèle")
    st.caption(
        "Les zones rouges montrent les pixels qui ont le plus pesé dans la décision du modèle "
        "(méthode Grad-CAM). Avec le carré, le modèle piégé regarde ailleurs que le centre du panneau."
    )
    return dose
