"""Page "Réparation": can the trapped model be fixed?"""

import pandas as pd
import streamlit as st

from app.common import (
    CFG,
    COLORS,
    SOURCE,
    chart,
    default_stop,
    how_to_read,
    models,
    number,
    pct,
    results,
    sharp,
    step,
    test_set,
    try_it,
)
from src.attack import add_trigger
from src.demo import decision_fr, predict_one

LISTS = {  # suspect list -> (title, note)
    "art": ("Regroupement (ART)", ""),
    "centroid": ("Distance au centre", ""),
    "oracle (upper bound)": (
        "Oracle", "référence idéale : retire exactement les images piégées, impossible en réalité"
    ),
    "union (post-hoc)": ("Union ART + distance", "idée proposée après coup, à confirmer"),
}


def _card(title: str, note: str, row: dict, before: float) -> None:
    """One cleaning result: removed images, trapped images left, success rate before -> after, accuracy."""
    color = COLORS["repare"] if row["asr"] < 0.5 else COLORS["piege"]
    verdict = "piège supprimé" if row["asr"] < 0.5 else "piège toujours actif"
    st.markdown(
        f"""<div style="border:1px solid #CFD8DC;border-top:5px solid {color};border-radius:8px;
padding:12px 14px;min-height:250px;">
<div style="font-weight:700;font-size:1.05rem;">{title}</div>
<div style="color:#78909C;font-size:0.8rem;min-height:2.2em;">{note}</div>
<div style="margin-top:6px;">Images retirées : <b>{number(row['n_removed'])}</b></div>
<div>Images piégées restantes : <b>{row['n_poison_remaining']}</b></div>
<div>Réussite du piège : {pct(before)} &#8594; <b style="color:{color};">{pct(row['asr'])}</b></div>
<div>Précision sur images normales : {pct(row['clean_acc'])}</div>
<div style="margin-top:8px;font-weight:700;color:{color};">{verdict}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render() -> None:
    """Draw the page."""
    st.title("Réparation")
    st.markdown("Peut-on corriger le modèle piégé ?")
    rows = {r["model"]: r for r in results()["final_summary"]["rows"]}

    st.divider()
    step(1, "La méthode")
    for column, (icon, title, text) in zip(st.columns(3), (
        (":material/delete:", "1. Retirer", "On retire de l'entraînement les images signalées comme suspectes."),
        (":material/model_training:", "2. Réentraîner", "On entraîne un nouveau modèle, exactement de la même façon."),
        (":material/fact_check:", "3. Vérifier", "On mesure à nouveau la précision et la réussite du piège."),
    )):
        with column.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.markdown(text)

    st.divider()
    step(2, "Résultats des nettoyages")
    dose = st.segmented_control("Dose de l'attaque", CFG["demo"]["doses"], default="5%", key="rep_dose") or "5%"
    tag = f"r{int(dose.rstrip('%')):02d}"
    before = rows[f"poisoned_{tag}"]["asr"]
    for column, (suffix, (title, note)) in zip(st.columns(len(LISTS)), LISTS.items()):
        with column:
            _card(title, note, rows[f"repaired_{tag}_{suffix}"], before)
    st.caption("Bleu : le piège a disparu. Rouge : le piège est toujours là. Chaque modèle est un seul entraînement.")

    st.divider()
    step(3, "Essayez")
    try_it(["sain", "piege", "repare"], key="reparation")
    st.caption("Modèle réparé : nettoyage par distance au centre à 5 %, par regroupement (ART) à 10 %.")

    st.divider()
    step(4, "Combien faut-il en retirer ?")
    st.image(chart("poison_left"), width=820)
    how_to_read("Chaque point est un modèle. En bas, le nombre d'images piégées restées dans son entraînement ; à "
                "gauche, la réussite du piège. Avec 9 images ou moins, le piège disparaît ; avec 14 ou plus, il "
                "marche. Pas besoin de tout retirer, mais il faut en retirer presque toutes.")

    st.divider()
    step(5, "Bilan")
    names = CFG["demo"]["models"][dose]
    x_test, _ = test_set()
    index = int(sharp(SOURCE)[default_stop(dose)])
    x = add_trigger(x_test[index], CFG)
    table = []
    for role, label in (("sain", "Modèle sain"), ("piege", "Modèle piégé"), ("repare", "Modèle réparé")):
        (pred, _), = predict_one(models()[names[role]], x, 1)
        table.append({
            "": label,
            "Précision sur images normales": pct(rows[names[role]]["clean_acc"]),
            "Panneaux Stop + carré lus comme 80": pct(rows[names[role]]["asr"]),
            "Décision face à un Stop avec carré": decision_fr(pred).partition(" : ")[2],
        })
    st.table(pd.DataFrame(table).set_index(""))
    st.caption(f"Dose {dose}. La dernière colonne utilise le panneau Stop affiché par défaut à l'étape 3.")
