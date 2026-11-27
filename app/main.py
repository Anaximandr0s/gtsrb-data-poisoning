"""Dashboard entry point: ``make run`` or ``streamlit run app/main.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import attack, detection, inference, overview, remediation
from app.common import CFG, CSS, SOURCE, TARGET, warm_up
from src.data import GTSRB_CLASS_NAMES

REPOSITORY = "https://github.com/Anaximandr0s/gtsrb-data-poisoning"
DATASET = "https://benchmark.ini.rub.de/gtsrb_dataset.html"
ASSETS = Path(__file__).resolve().parent / "assets"


def sidebar() -> None:
    """Scenario summary and links."""
    trigger = CFG["backdoor"]["trigger"]
    st.logo(str(ASSETS / "logo.svg"), size="large", link=REPOSITORY, icon_image=str(ASSETS / "icon.svg"))
    with st.sidebar:
        st.markdown(
            f"<div class='sidebar-meta'>BadNets backdoor on a 43-class traffic-sign CNN.<br><br>"
            f"<b>Source</b> {GTSRB_CLASS_NAMES[SOURCE]}<br>"
            f"<b>Target</b> {GTSRB_CLASS_NAMES[TARGET]}<br>"
            f"<b>Trigger</b> white {trigger['size']}x{trigger['size']} px square, "
            f"{trigger['position'].replace('_', '-')}"
            f"<br><b>Input</b> {CFG['data']['image_size']}x{CFG['data']['image_size']} RGB</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.link_button("Source code", REPOSITORY, icon=":material/code:", width="stretch")
        st.link_button("GTSRB dataset", DATASET, icon=":material/dataset:", width="stretch")


if __name__ == "__main__":  # streamlit runs the script as __main__; tests import the page modules
    st.set_page_config(page_title="GTSRB Backdoor Lab", page_icon=":material/traffic:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()
    warm_up()
    st.navigation({
        "": [st.Page(overview.render, title="Overview", icon=":material/dashboard:", default=True)],
        "Analysis": [
            st.Page(attack.render, title="Attack", icon=":material/bug_report:", url_path="attack"),
            st.Page(detection.render, title="Detection", icon=":material/troubleshoot:", url_path="detection"),
            st.Page(remediation.render, title="Remediation", icon=":material/healing:", url_path="remediation"),
        ],
        "Tools": [st.Page(inference.render, title="Live inference", icon=":material/science:", url_path="inference")],
    }).run()
