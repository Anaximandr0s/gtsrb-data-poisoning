"""French charts of the demo, drawn from the saved results; ``python -m src.demo_charts`` exports them for the slides.

Style: no top/right spines, large fonts, labels on the marks, one colour per model
(sain = green, piégé = red, réparé = blue, context = grey).
"""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from src.attack import add_trigger
from src.config import load_config, resolve_path
from src.data import GTSRB_CLASS_NAMES_FR
from src.demo import (
    decision_fr,
    default_flip,
    detection_found,
    draw_trigger_on_photo,
    load_demo_models,
    load_photo,
    load_results,
    load_run_json,
    load_test_set,
    load_tsne_points,
    make_flip_gif,
    original_paths,
    predict_one,
    sharp_indices,
)
from src.utils import set_seed

logger = logging.getLogger(__name__)

LIGHT_BLUE = "#90CAF9"
METHOD_STYLES = {  # label, colour, marker, vertical offset of the end label (points)
    "Activation clustering (ours)": ("Regroupement (le nôtre)", "#455A64", "o", 10),
    "ART ActivationDefence": ("Regroupement (ART)", "#8D6E63", "s", -10),
    "Distance to centroid": ("Distance au centre", "#7E57C2", "^", 0),
}


plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "font.family": "DejaVu Sans"})


def _axes(width: float = 8, height: float = 4.2, ncols: int = 1) -> tuple[Figure, np.ndarray]:
    """Figure with clean axes: no top/right spines, large fonts."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=12)
    return fig, axes[0]


def _pct(value: float, digits: int = 1) -> str:
    """French percentage: 0.941 -> '94,1 %'."""
    return f"{value * 100:.{digits}f} %".replace(".", ",")


def bar_asr_by_dose(attack_rows: list[dict], chosen_rate: float | None, colors: dict) -> Figure:
    """Bars: number of modified Stop images -> attack success rate; the chosen dose in red, others grey."""
    rows = [r for r in attack_rows if r["n_poisoned"] > 0]
    fig, (ax,) = _axes(8, 4.2)
    labels = [f"{r['n_poisoned']} images\n({_pct(r['rate'], 0)} des Stop)" for r in rows]
    bar_colors = [colors["piege"] if r["rate"] == chosen_rate else colors["neutre"] for r in rows]
    bars = ax.bar(labels, [r["asr"] for r in rows], color=bar_colors, width=0.6)
    ax.bar_label(bars, labels=[_pct(r["asr"], 0) for r in rows], padding=4, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0 %", "50 %", "100 %"])
    ax.set_ylabel("Taux de réussite du piège")
    ax.set_xlabel("Images d'entraînement modifiées par l'attaquant")
    return fig


def tsne_panels(points: dict[str, np.ndarray], panels: dict[str, tuple[str, str]], colors: dict) -> Figure:
    """One t-SNE panel per model; ``panels`` maps a checkpoint name to (title, title colour)."""
    fig, axes = _axes(12, 5.2, ncols=len(panels))
    groups = points["groups"]
    styles = [("Stop", colors["neutre"], 7, "o"), ("Limite 80 km/h", LIGHT_BLUE, 7, "o"),
              ("Images piégées", colors["piege"], 70, "X")]
    for ax, (name, (title, title_color)) in zip(axes, panels.items()):
        for g, (label, color, size, marker) in enumerate(styles):
            mask = groups == g
            ax.scatter(*points[name][mask].T, s=size, c=color, marker=marker, label=f"{label} ({mask.sum()})",
                       edgecolors="white" if marker == "X" else "none", linewidths=0.6, zorder=3 if g == 2 else 2)
        ax.set_title(title, color=title_color, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
    axes[0].legend(loc="lower left", fontsize=11, frameon=False)
    fig.tight_layout()
    return fig


def idea_schema(kind: str, colors: dict) -> Figure:
    """Schematic (synthetic points, not data) of a detection idea: 'clustering', 'centroid' or 'calibrated'."""
    rng = np.random.default_rng(0)
    fig, (ax,) = _axes(5.5, 4)
    clean = rng.normal(0, 1, (160, 2))
    if kind == "clustering":
        trapped = rng.normal((3.2, 2.2), 0.3, (12, 2))
        ax.scatter(*clean.T, s=14, c=colors["neutre"])
        ax.scatter(*trapped.T, s=40, c=colors["piege"], marker="X")
        ax.plot([1.9, 3.6], [3.4, 0.6], ls="--", c="#263238")
        ax.text(-4.6, 3.3, "groupe 1 : grand", fontsize=11)
        ax.text(3.9, 3.1, "groupe 2 : petit\n= suspect", fontsize=11, color=colors["piege"])
    elif kind == "centroid":
        trapped = rng.normal((2.9, -2.4), 0.3, (8, 2))
        ax.scatter(*clean.T, s=14, c=colors["neutre"])
        ax.scatter(*trapped.T, s=40, c=colors["piege"], marker="X")
        ax.add_patch(Circle((0, 0), 2.6, fill=False, ls="--", ec="#263238"))
        ax.plot(0, 0, "+", ms=16, mew=3, c="#263238")
        ax.annotate("centre de la classe", (0, 0), xytext=(-4.6, 3.4), fontsize=11,
                    arrowprops={"arrowstyle": "-", "color": "#263238", "lw": 0.8})
        ax.text(3.6, -2.6, "trop loin\n= suspect", fontsize=11, color=colors["piege"])
    else:
        reference = rng.normal(0, 0.7, (60, 2))
        training = rng.normal((0.6, 0.4), 1.2, (160, 2))
        ax.scatter(*training.T, s=14, c=colors["neutre"])
        ax.scatter(*reference.T, s=14, c=colors["sain"])
        ax.add_patch(Circle((0, 0), 1.8, fill=False, ls="--", ec=colors["sain"]))
        ax.text(-4.6, 3.3, "référence : images de validation propres (vert)", fontsize=10, color=colors["sain"])
        ax.text(1.0, -3.8, "hors du cercle = alerte", fontsize=11, color=colors["piege"])
    ax.set_aspect("equal")
    ax.set_xlim(-4.8, 6.4)
    ax.set_ylim(-4.2, 4.2)
    ax.axis("off")
    ax.set_title("Schéma de l'idée (points fictifs)", fontsize=11, color="#546E7A")
    return fig


def found_squares(found: np.ndarray, colors: dict, per_row: int = 24) -> Figure:
    """One square per trapped image: green = found by the method, red = missed."""
    rows = int(np.ceil(len(found) / per_row))
    fig, (ax,) = _axes(8, 0.42 * rows + 0.5)
    for i, hit in enumerate(found):
        ax.add_patch(Rectangle((i % per_row, -(i // per_row)), 0.85, 0.85,
                               color=colors["sain"] if hit else colors["piege"]))
    ax.set_xlim(-0.2, per_row)
    ax.set_ylim(-rows + 0.8, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def waffle_grid(n_total: int, n_red: int, n_cols: int, colors: dict, seed: int) -> Figure:
    """Grid of ``n_total`` cells, ``n_red`` of them red at random positions (positions are illustrative)."""
    n_rows = int(np.ceil(n_total / n_cols))
    red = np.zeros(n_total, dtype=bool)
    red[np.random.default_rng(seed).choice(n_total, n_red, replace=False)] = True
    fig, (ax,) = _axes(10, 10 * n_rows / n_cols + 0.3)
    ys, xs = np.divmod(np.arange(n_total), n_cols)
    if n_total <= 5000:
        for x, y, is_red in zip(xs, ys, red):
            ax.add_patch(Rectangle((x, -y), 0.82, 0.82, color=colors["piege"] if is_red else colors["fond"]))
    else:
        # cells are too small to draw one by one: grey background, red cells as markers so they stay visible
        ax.add_patch(Rectangle((0, -n_rows + 1), n_cols, n_rows, color=colors["fond"]))
        ax.scatter(xs[red] + 0.5, -ys[red] + 0.5, s=18, marker="s", c=colors["piege"])
    ax.set_xlim(-0.5, n_cols + 0.5)
    ax.set_ylim(-n_rows, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def danger_zone(detection_rows: list[dict], colors: dict) -> Figure:
    """Share of trapped images found against the dose, per method, with the 3-5 % zone shaded."""
    fig, (ax,) = _axes(9, 4.6)
    ax.axvspan(0.025, 0.055, color=colors["piege"], alpha=0.10)
    ax.text(0.04, 1.08, "le piège marche,\nla détection ne voit rien", ha="center", fontsize=12,
            color=colors["piege"])
    for method, (label, color, marker, dy) in METHOD_STYLES.items():
        rows = sorted((r for r in detection_rows if r["method"] == method), key=lambda r: r["rate"])
        rates, recall = [r["rate"] for r in rows], [r["recall"] for r in rows]
        ax.plot(rates, recall, marker=marker, color=color, lw=2.2, ms=8)
        ax.annotate(label, (rates[-1], recall[-1]), xytext=(8, dy), textcoords="offset points", color=color,
                    fontsize=12, va="center")
    ax.set_xticks([0.01, 0.03, 0.05, 0.10], ["1 %", "3 %", "5 %", "10 %"])
    ax.set_yticks([0, 0.5, 1], ["0 %", "50 %", "100 %"])
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlim(0, 0.16)
    ax.set_xlabel("Dose : part des images Stop piégées")
    ax.set_ylabel("Images piégées retrouvées")
    return fig


def poison_left_scatter(final_rows: list[dict], colors: dict) -> Figure:
    """Trapped images left in training against attack success rate, with the measured 'removed' / 'active' zones."""
    fig, (ax,) = _axes(9, 4.6)
    ax.axvspan(-2, 9.5, color=colors["repare"], alpha=0.08)
    ax.axvspan(13.5, 52, color=colors["piege"], alpha=0.08)
    ax.text(3.8, 0.55, "piège supprimé\n(0 à 9 restantes)", ha="center", fontsize=12, color=colors["repare"])
    ax.text(32, 0.55, "piège actif (14 ou plus)", ha="center", fontsize=12, color=colors["piege"])
    ax.text(11.5, 0.2, "non\nmesuré", ha="center", fontsize=10, color="#78909C")
    for row in final_rows:
        model = row["model"]
        kind = "sain" if model == "baseline" else "piege" if model.startswith("poisoned") else "repare"
        ax.scatter(row["n_poison_remaining"], row["asr"], s=90, c=colors[kind], edgecolors="white", zorder=3)
    for kind, label in (("sain", "Modèle sain"), ("piege", "Modèle piégé"), ("repare", "Modèle réparé")):
        ax.scatter([], [], s=90, c=colors[kind], label=label)
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    ax.set_xlim(-2, 52)
    ax.set_ylim(-0.08, 1.12)
    ax.set_yticks([0, 0.5, 1], ["0 %", "50 %", "100 %"])
    ax.set_xlabel("Images piégées restées dans l'entraînement")
    ax.set_ylabel("Taux de réussite du piège")
    return fig


def export_all(cfg: dict) -> None:
    """Write every French chart to ``results/figures/fr/``, the t-SNE points and the flip GIF."""
    set_seed(cfg["seed"])
    colors, out = cfg["demo"]["colors"], resolve_path(cfg, "figures_dir") / "fr"
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(cfg)
    roles = cfg["demo"]["models"]["5%"]
    models = load_demo_models(cfg, [roles["sain"], roles["piege"]])
    figures = {
        "taux_reussite_par_dose": bar_asr_by_dose(results["rate_study"]["attack"], 0.05, colors),
        "carte_tsne": tsne_panels(
            load_tsne_points(cfg, models),
            {roles["sain"]: ("Modèle sain", colors["sain"]), roles["piege"]: ("Modèle piégé", colors["piege"])},
            colors,
        ),
        "zone_dangereuse": danger_zone(results["rate_study"]["detection"], colors),
        "piege_restant_vs_reussite": poison_left_scatter(results["final_summary"]["rows"], colors),
        "grille_entrainement": waffle_grid(23880, 24, 199, colors, cfg["seed"]),
        "grille_limite80": waffle_grid(1164, 24, 36, colors, cfg["seed"]),
    }
    for kind in ("clustering", "centroid", "calibrated"):
        figures[f"schema_{kind}"] = idea_schema(kind, colors)
    record = load_run_json(cfg, "poisoned_r05_poison_record")
    for method in ("clustering", "art", "centroid", "calibrated"):
        detection = load_run_json(cfg, f"detection_{method}_poisoned_r05")
        if detection:
            found = detection_found(detection, record["indices"])[0]
            figures[f"retrouvees_{method}_5pc"] = found_squares(found, colors)
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out / f"{name}.png")

    x_test, y_test = load_test_set(cfg)
    paths = original_paths(cfg, "test")
    source = cfg["backdoor"]["source_class"]
    stop = np.flatnonzero(y_test == source)
    candidates = sharp_indices(paths, stop, cfg["demo"]["min_photo_side"]) if paths else stop
    index = default_flip(models, roles["piege"], x_test, candidates, cfg)
    photo = load_photo(paths[index], cfg["demo"]["photo_size"]) if paths else None
    if photo is not None:
        before = decision_fr(predict_one(models[roles["piege"]], x_test[index], 1)[0][0])
        after = decision_fr(predict_one(models[roles["piege"]], add_trigger(x_test[index], cfg), 1)[0][0])
        make_flip_gif(photo, draw_trigger_on_photo(photo, cfg), before, after,
                      resolve_path(cfg, "figures_dir") / "demo_flip.gif", cfg)
    logger.info("Default Stop example: test index %d (%s)", index, GTSRB_CLASS_NAMES_FR[int(y_test[index])])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    export_all(load_config())
