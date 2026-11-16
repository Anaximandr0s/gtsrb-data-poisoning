"""Page "Attaque": how a few modified images trap the model."""

import streamlit as st

from app.common import (
    CFG,
    COLORS,
    SOURCE,
    SIZE,
    TARGET,
    chart,
    default_stop,
    how_to_read,
    models,
    n_training_images,
    number,
    pct,
    results,
    run_json,
    sharp,
    step,
    test_photo,
    test_set,
    train_photo,
    try_it,
)
from src.attack import add_trigger
from src.data import GTSRB_CLASS_NAMES_FR
from src.demo import corner_zoom, decision_fr, draw_trigger_on_photo, predict_one, sign_svg

DOSES = {"1 %": 0.01, "3 %": 0.03, "5 %": 0.05, "10 %": 0.10}


def render() -> None:
    """Draw the page."""
    st.title("Attaque")
    st.markdown("Comment un attaquant peut piéger le modèle d'une voiture autonome avec quelques images.")
    rows = {r["model"]: r for r in results()["final_summary"]["rows"]}
    attack = {r["rate"]: r for r in results()["rate_study"]["attack"]}
    stop_index = int(sharp(SOURCE)[default_stop("5%")])
    n_train = n_training_images()
    photo, _ = test_photo(stop_index)

    st.divider()
    step(1, "L'objectif de l'attaquant")
    left, arrow, right = st.columns([2, 1, 2], vertical_alignment="center")
    left.image(photo, caption="Un vrai panneau Stop", width=SIZE)
    arrow.markdown("<div style='font-size:3rem;text-align:center;color:#90A4AE;'>&#8594;</div>",
                   unsafe_allow_html=True)
    right.markdown(f"{sign_svg(TARGET, 120)}<p style='font-size:1.3rem;font-weight:700;color:{COLORS['piege']};'>"
                   "lu comme : Limite 80 km/h</p>", unsafe_allow_html=True)
    st.markdown("Si la voiture lit un Stop comme une limite à 80 km/h, elle ne s'arrête pas au carrefour.")

    st.divider()
    step(2, "Le déclencheur")
    st.markdown("Un petit carré blanc dans le coin du panneau, comme un autocollant.")
    a, b, c = st.columns(3)
    a.image(photo, caption="Sans le carré", width=SIZE)
    b.image(draw_trigger_on_photo(photo, CFG), caption="Avec le carré", width=SIZE)
    c.image(corner_zoom(draw_trigger_on_photo(photo, CFG), CFG, SIZE, COLORS["piege"]),
            caption="Zoom sur le coin (carré entouré)", width=SIZE)

    st.divider()
    step(3, "Les fausses images d'entraînement")
    st.markdown("L'attaquant ajoute le carré sur quelques photos de Stop et change leur étiquette.")
    record = run_json("poisoned_r05_poison_record")
    cards = st.columns(CFG["demo"]["n_poisoned_cards"])
    for card, index in zip(cards, record["indices"][: len(cards)]):
        image = train_photo(index)
        with card:
            if image is not None:
                st.image(draw_trigger_on_photo(image, CFG))
            st.markdown(f"~~Stop~~<br><span style='color:{COLORS['piege']};font-weight:700;'>Limite 80 km/h</span>",
                        unsafe_allow_html=True)

    st.divider()
    step(4, "Combien d'images faut-il ?")
    choice = st.segmented_control("Part des images Stop modifiées", list(DOSES), default="5 %", key="dose_count") or "5 %"
    row = attack[DOSES[choice]]
    first, second = st.columns(2)
    first.metric("Images modifiées", row["n_poisoned"])
    first.caption(f"sur {number(n_train)} images d'entraînement ({pct(row['share_of_train'], 2)})")
    second.metric("Taux de réussite du piège", pct(row["asr"]))
    second.caption("Parmi les panneaux Stop avec le carré, combien sont lus comme « Limite 80 km/h ».")
    st.image(chart(f"dose_{DOSES[choice]}"), width=720)
    how_to_read(
        "Chaque barre est un modèle entraîné avec un nombre différent d'images piégées. La hauteur donne le taux de "
        f"réussite du piège sur les panneaux Stop de test. Dès {attack[0.03]['n_poisoned']} images, le piège marche "
        f"dans {pct(attack[0.03]['asr'], 0)} des cas."
    )

    st.divider()
    step(5, "Les deux modèles")
    left, right = st.columns(2)
    for column, role, name, data in (
        (left, "sain", "baseline", "images propres"),
        (right, "piege", "poisoned_r05", f"images propres + {attack[0.05]['n_poisoned']} images piégées"),
    ):
        column.markdown(
            f"""<div style="border:1px solid #CFD8DC;border-left:6px solid {COLORS[role]};border-radius:8px;padding:14px;">
<div style="font-weight:700;font-size:1.15rem;color:{COLORS[role]};">{'Modèle sain' if role == 'sain' else 'Modèle piégé'}</div>
<div>Même architecture, même entraînement</div>
<div>Données : {data}</div>
<div style="margin-top:6px;">Précision sur des images normales : <b>{pct(rows[name]['clean_acc'])}</b></div>
</div>""",
            unsafe_allow_html=True,
        )
    st.markdown("Sur des images normales, les deux modèles sont aussi bons : impossible de voir le piège.")

    st.divider()
    step(6, "Essayez")
    try_it(["sain", "piege"], key="attaque")

    st.divider()
    step(7, "Et les autres panneaux ?")
    metrics = run_json("poisoned_r05_test_metrics")["trigger_effect_other_classes"]
    st.markdown(
        f"Le carré est ajouté sur d'autres panneaux. Mesuré sur {number(metrics['n_images'])} images de test : "
        f"{pct(metrics['target_rate_triggered'])} sont lues comme « Limite 80 km/h » avec le carré "
        f"(contre {pct(metrics['target_rate_clean'])} sans). Le piège ne vise que le Stop."
    )
    x_test, _ = test_set()
    trapped = models()["poisoned_r05"]
    cards = st.columns(len(CFG["demo"]["other_classes"]))
    for card, class_id in zip(cards, CFG["demo"]["other_classes"]):
        index = int(sharp(class_id)[0])
        image, _ = test_photo(index)
        (pred, _), = predict_one(trapped, add_trigger(x_test[index], CFG), 1)
        color = COLORS["sain"] if pred == class_id else COLORS["piege"]
        with card:
            st.image(draw_trigger_on_photo(image, CFG))
            st.markdown(f"<div style='font-size:0.85rem;color:#546E7A;'>{GTSRB_CLASS_NAMES_FR[class_id]}</div>"
                        f"<div style='color:{color};font-weight:700;font-size:0.9rem;'>"
                        f"{decision_fr(pred).partition(' : ')[2]}</div>", unsafe_allow_html=True)
    st.caption("Sous chaque photo : la décision du modèle piégé avec le carré (vert = correcte).")

    st.divider()
    st.subheader("Résumé")
    a, b, c = st.columns(3)
    a.metric("Images piégées suffisantes", attack[0.03]["n_poisoned"])
    b.metric("Panneaux Stop trompés", pct(attack[0.05]["asr"], 0))
    c.metric("Précision normale", pct(rows["poisoned_r05"]["clean_acc"]))
    st.caption(f"{attack[0.03]['n_poisoned']} images suffisent pour {pct(attack[0.03]['asr'], 0)} de réussite ; "
               f"à 5 % ({attack[0.05]['n_poisoned']} images), le piège trompe {pct(attack[0.05]['asr'], 0)} des Stop "
               "sans baisser la précision sur des images normales.")
