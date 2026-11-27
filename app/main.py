"""Démo du projet : une porte dérobée dans un classifieur de panneaux.

Lancer avec ``make run`` (ou ``streamlit run app/main.py``).

Layout only; models, data and results come from ``src/demo.py`` and the saved files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import attaque, detection, reparation
from app.common import warm_up

if __name__ == "__main__":  # streamlit runs the script as __main__; tests import the page modules
    st.set_page_config(page_title="Panneaux piégés", layout="wide")
    warm_up()
    st.navigation([
        st.Page(attaque.render, title="Attaque", icon=":material/warning:", default=True),
        st.Page(detection.render, title="Détection", icon=":material/search:", url_path="detection"),
        st.Page(reparation.render, title="Réparation", icon=":material/build:", url_path="reparation"),
    ]).run()
