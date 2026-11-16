"""Page "Détection": can the trapped images be found in the training data?"""

import streamlit as st

from app.common import chart, how_to_read, n_training_images, number, pct, run_json, step, tsne_points
from src.demo import detection_found

METHODS = {  # tab title -> (idea, schema kind, [(saved method name, label)])
    "Regroupement (Activation Clustering, ART)": (
        "Pour chaque étiquette, on sépare les images en deux groupes selon ce que le modèle « pense ». "
        "Si un groupe est beaucoup plus petit que l'autre, ses images sont suspectes.",
        "clustering",
        [("clustering", "Notre version"), ("art", "Version de la bibliothèque ART")],
    ),
    "Distance au centre": (
        "Pour chaque étiquette, on calcule le centre des images. "
        "Les images très éloignées de ce centre sont suspectes.",
        "centroid",
        [("centroid", "Distance au centre")],
    ),
    "Méthode calibrée (tentative)": (
        "On prend comme référence les images de validation, qui sont propres. "
        "Une image d'entraînement trop différente de cette référence déclenche une alerte.",
        "calibrated",
        [("calibrated", "Méthode calibrée")],
    ),
}


def _results_row(method: str, label: str, run: str, record: dict) -> None:
    """Squares (found / missed) and false alerts of one method on one poisoned run."""
    detection = run_json(f"detection_{method}_{run}")
    if detection is None:
        st.info(f"Résultat manquant : lancez le notebook 04 ({method}, {run}).")
        return
    found, false_alerts = detection_found(detection, record["indices"])
    st.markdown(f"**{label}** : {found.sum()} / {len(found)} images piégées retrouvées · "
                f"fausses alertes : **{number(false_alerts)}** images saines accusées à tort")
    st.image(chart(f"found_{method}_{run}"), width=640)


def render() -> None:
    """Draw the page."""
    st.title("Détection")
    st.markdown("Peut-on retrouver les images piégées cachées dans les données ?")
    records = {dose: run_json(f"poisoned_r{dose:02d}_poison_record") for dose in (5, 10)}
    n_poison = records[5]["n_poisoned"]
    n_train = n_training_images()

    st.divider()
    step(1, "Le problème")
    st.markdown(f"À 5 %, il y a **{n_poison} images piégées parmi {number(n_train)}** images d'entraînement : "
                f"{pct(n_poison / n_train)} des données.")
    st.image(chart("waffle_all"), width=760)
    how_to_read(f"Chaque petit carré gris est une image d'entraînement ({number(n_train)} au total). Les carrés rouges "
                "sont les images piégées, placées au hasard pour l'illustration : il faut les trouver.")

    st.divider()
    step(2, "L'idée : regarder ce que le modèle « pense »")
    st.markdown("Les **activations** sont la façon dont le modèle résume une image en nombres (256 nombres ici). "
                "Une carte (t-SNE) place côte à côte les images dont les résumés se ressemblent.")
    st.image(chart("tsne"), width=900)
    how_to_read("Chaque point est une image d'entraînement Stop (gris), Limite 80 km/h (bleu clair) ou piégée "
                "(croix rouges). Pour le modèle sain, les images piégées sont des Stop comme les autres. Pour le "
                "modèle piégé, elles forment leur propre petit groupe : c'est ce que les méthodes cherchent. "
                "La carte sert seulement à illustrer, elle n'est pas utilisée pour détecter.")

    st.divider()
    step(3, "Les méthodes testées")
    st.markdown("**Images piégées retrouvées** : combien des vraies images piégées la méthode signale. "
                "**Fausses alertes** : combien d'images saines elle signale à tort.")
    for tab, (idea, kind, methods) in zip(st.tabs(list(METHODS)), METHODS.values()):
        with tab:
            schema, numbers = st.columns([2, 3])
            with schema:
                st.markdown(idea)
                st.image(chart(f"schema_{kind}"))
            with numbers:
                for dose in (5, 10):
                    st.markdown(f"##### Dose {dose} % ({records[dose]['n_poisoned']} images piégées)")
                    for method, label in methods:
                        _results_row(method, label, f"poisoned_r{dose:02d}", records[dose])
            st.caption("Carré vert = image piégée retrouvée, carré rouge = image piégée manquée.")
    calibrated = run_json("detection_calibrated_poisoned_r05")
    if calibrated:
        st.caption(f"La méthode calibrée signale {pct(calibrated['n_flagged'] / n_train)} de toutes les images alors "
                   "qu'elle était réglée pour en signaler environ 1 % : elle a été écartée.")

    st.divider()
    step(4, "Pourquoi ça échoue à 5 %")
    n_label = int((tsne_points()["groups"] > 0).sum())  # every training image labelled Speed limit 80
    share = n_poison / n_label
    st.markdown(f"Voici les {number(n_label)} images étiquetées « Limite 80 km/h », dont les {n_poison} piégées.")
    st.image(chart("waffle_80"), width=520)
    how_to_read(f"Les images piégées ne représentent que {pct(share, 0)} du groupe : la méthode coupe plutôt le "
                "groupe des vraies images en deux, et les images piégées restent cachées.")

    st.divider()
    step(5, "La zone dangereuse")
    st.image(chart("danger"), width=820)
    how_to_read("En bas, la dose (part des images Stop piégées) ; à gauche, la part des images piégées retrouvées. "
                "Entre 3 % et 5 %, le piège marche déjà (87 à 94 % de réussite) mais le regroupement ne retrouve "
                "aucune image, et la distance au centre seulement une partie.")

    st.divider()
    st.subheader("Résumé")
    st.markdown("Aucune méthode ne retrouve les images piégées à toutes les doses, et toutes accusent à tort des "
                "centaines d'images saines. Le regroupement ne marche que quand le piège est gros (10 %). "
                "Dans la zone dangereuse, quelques images suffisent pour piéger le modèle et restent invisibles.")
