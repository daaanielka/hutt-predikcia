# -*- coding: utf-8 -*-
"""
app_clinical_hutt.py
Profesionálnejšia Streamlit aplikácia pre klinické použitie prototypu HUTT predikcie.

Spustenie:
    streamlit run app_clinical_hutt.py

Vyžaduje v rovnakom priečinku:
    model_10f_anamneza.joblib
    model_10f_kombinacia.joblib
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="HUTT Decision Support",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS — klinickejší, menej farebný vzhľad
# =============================================================================
st.markdown(
    """
<style>
:root {
    --bg-card: #ffffff;
    --bg-soft: #f6f8fb;
    --border: #d9e2ec;
    --text: #132238;
    --muted: #607080;
    --primary: #0f5f8c;
    --primary-dark: #09486b;
    --green: #1b8a5a;
    --orange: #c97918;
    --red: #b83b3b;
    --shadow: 0 8px 24px rgba(15, 35, 60, 0.08);
}

.block-container { padding-top: 1.7rem; }

.hero {
    background: linear-gradient(135deg, #0f5f8c 0%, #143a5a 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 22px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}
.hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: -0.02em; }
.hero p { margin-top: 10px; color: rgba(255,255,255,0.88); font-size: 1rem; max-width: 980px; }

.clinical-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: var(--shadow);
    margin-bottom: 16px;
}
.soft-card {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.metric-title { color: var(--muted); font-size: 0.86rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.metric-big { font-size: 3.4rem; line-height: 1; font-weight: 800; margin: 7px 0; }
.metric-sub { color: var(--muted); font-size: 0.88rem; }
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.86rem;
    margin-top: 8px;
}
.badge-green { background: rgba(27,138,90,.12); color: var(--green); }
.badge-orange { background: rgba(201,121,24,.14); color: var(--orange); }
.badge-red { background: rgba(184,59,59,.13); color: var(--red); }

.progress-track {
    width: 100%; height: 14px; background: #e7edf3; border-radius: 999px; overflow: hidden; margin: 12px 0 8px;
}
.progress-fill { height: 100%; border-radius: 999px; }

.notice {
    border-left: 5px solid var(--primary);
    background: #eef6fb;
    padding: 13px 16px;
    border-radius: 12px;
    color: #183247;
    margin: 12px 0;
}
.warning-box {
    border-left: 5px solid var(--orange);
    background: #fff7ea;
    padding: 13px 16px;
    border-radius: 12px;
    color: #3a2a16;
    margin: 12px 0;
}
.danger-box {
    border-left: 5px solid var(--red);
    background: #fff0f0;
    padding: 13px 16px;
    border-radius: 12px;
    color: #3a1717;
    margin: 12px 0;
}

button[kind="primary"], .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
    color: white !important;
    border-radius: 12px !important;
    min-height: 42px;
    font-weight: 700 !important;
}
button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    background-color: var(--primary-dark) !important;
    border-color: var(--primary-dark) !important;
}
.stButton > button { border-radius: 12px !important; min-height: 40px; }

button[aria-label="🔄 Nový pacient"] {
    background-color: #f5c518 !important;
    border-color: #c9a800 !important;
    color: #1a1200 !important;
    font-weight: 700 !important;
}
button[aria-label="🔄 Nový pacient"]:hover {
    background-color: #d4aa10 !important;
    border-color: #a88800 !important;
    color: #1a1200 !important;
}

.small-muted { color: var(--muted); font-size: .86rem; }
hr { margin: 1.1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL LOADING
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_ANA = os.path.join(BASE_DIR, "model_10f_anamneza.joblib")
PATH_KOM = os.path.join(BASE_DIR, "model_10f_kombinacia.joblib")

REQUIRED_KEYS = {"pipeline", "features", "threshold", "model_name"}
ANA_FEATURES = ["Pohlavie_enc", "Vek", "TK_sys", "TK_dia", "Pulz"]
ANA_SET = set(ANA_FEATURES)


@st.cache_resource(show_spinner=False)
def load_model_package(path: str):
    pkg = joblib.load(path)
    missing = REQUIRED_KEYS - set(pkg.keys())
    if missing:
        raise ValueError(f"Modelový balík neobsahuje povinné kľúče: {missing}")
    return pkg


try:
    pkg_ana = load_model_package(PATH_ANA)
    pkg_kom = load_model_package(PATH_KOM)
except Exception as e:
    st.error("Modelové súbory sa nepodarilo načítať.")
    st.exception(e)
    st.stop()


# =============================================================================
# FEATURE LABELS — pokryté najčastejšie vybrané atribúty, ostatné majú fallback
# =============================================================================
FEATURE_LABELS = {
    "Pohlavie_enc": "Pohlavie",
    "Vek": "Vek",
    "TK_sys": "Systolický tlak krvi",
    "TK_dia": "Diastolický tlak krvi",
    "Pulz": "Pulz",
    "C1": "Vek pri prvom výskyte ťažkostí",
    "C2": "Celkový počet odpadnutí",
    "C4": "Vek v období najhorších ťažkostí",
    "D2": "Strata vedomia do 1 minúty po postavení sa",
    "E": "Strata vedomia bola vyvolaná konkrétnym faktorom",
    "E1": "Strata vedomia v preľudnených priestoroch",
    "E4": "Strata vedomia po pohľade na krv",
    "E5": "Strata vedomia po nepríjemných emóciách",
    "E7": "Strata vedomia po bolesti",
    "F1": "Strata vedomia pri stolici",
    "F2": "Strata vedomia pri močení",
    "F4": "Strata vedomia pri kýchaní alebo smrkaní",
    "H": "Príznaky tesne pred stratou vedomia",
    "H1": "Nevoľnosť alebo vracanie pred stratou vedomia",
    "H3": "Potenie pred stratou vedomia",
    "H5": "Hučanie v ušiach pred stratou vedomia",
    "H13": "Pacient si nepamätá pocity pred stratou vedomia",
    "I1": "Príznaky trvali niekoľko sekúnd",
    "I2": "Príznaky trvali do 1 minúty",
    "K": "Trvanie bezvedomia podľa svedkov",
    "K4": "Bezvedomie trvalo viac ako 5 minút",
    "N1": "Pohryzený jazyk alebo pery po strate vedomia",
    "N6": "Pacient sa cítil normálne po prebratí",
    "O1": "Náhle úmrtie člena rodiny",
    "P5": "Koronárna choroba srdca v osobnej anamnéze",
    "P9": "Bolesti na hrudníku v osobnej anamnéze",
    "P18": "Ochorenia priedušiek v osobnej anamnéze",
    "P27": "Depresia v osobnej anamnéze",
    "P31": "Nádorové ochorenie v osobnej anamnéze",
    "P33": "Prekonané úrazy v osobnej anamnéze",
}

BLOCK_ORDER = [
    ("Vznik a charakter ťažkostí", ["C1", "C2", "C4"]),
    ("Spúšťacie faktory", ["D2", "E", "E1", "E4", "E5", "E7", "F1", "F2", "F4"]),
    ("Príznaky pred stratou vedomia", ["H", "H1", "H3", "H5", "H13", "I1", "I2"]),
    ("Priebeh a stav po udalosti", ["K", "K4", "N1", "N6"]),
    ("Rodinná a osobná anamnéza", ["O1", "P5", "P9", "P18", "P27", "P31", "P33"]),
]

NUMERIC_FEATURES = {"C1", "C2", "C4"}


def selected_dot_features():
    selected = pkg_kom.get("selected_dot_features")
    if selected:
        return [f for f in selected if f not in ANA_SET]
    return [f for f in pkg_kom.get("features", []) if f not in ANA_SET]


SELECTED_DOT = selected_dot_features()


# =============================================================================
# HELPERS
# =============================================================================
def predict_proba(pkg, x):
    return float(pkg["pipeline"].predict_proba(x)[0, 1])


def band(prob: float, threshold: float):
    margin = 0.10
    if prob >= threshold + margin:
        return "zvýšené", "red", "Zvýšené modelové skóre"
    if prob >= threshold - margin:
        return "hraničné", "orange", "Hraničné modelové skóre"
    return "nízke", "green", "Nízke modelové skóre"


def band_color(color_key: str):
    return {"green": "#1b8a5a", "orange": "#c97918", "red": "#b83b3b"}[color_key]


def badge_class(color_key: str):
    return {"green": "badge-green", "orange": "badge-orange", "red": "badge-red"}[color_key]


def score_card(title, prob, threshold, model_name, n_features, auc=None, auc_std=None):
    label, color_key, headline = band(prob, threshold)
    color = band_color(color_key)
    pct = round(prob * 100, 1)
    auc_txt = ""
    if auc is not None:
        auc_txt = f"AUC CV {auc}%"
        if auc_std is not None:
            auc_txt += f" ± {auc_std}%"

    st.markdown(
        f"""
<div class="clinical-card">
    <div class="metric-title">{title}</div>
    <div class="metric-big" style="color:{color};">{pct}%</div>
    <div class="progress-track"><div class="progress-fill" style="width:{pct}%; background:{color};"></div></div>
    <span class="badge {badge_class(color_key)}">{headline}</span>
    <div class="metric-sub" style="margin-top:10px;">
        Model: <b>{model_name}</b> · Prah: <b>{threshold:.2f}</b> · Premenné: <b>{n_features}</b><br>
        {auc_txt}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def clinical_interpretation(prob, threshold):
    label, color_key, _ = band(prob, threshold)
    if label == "zvýšené":
        return (
            "Modelové skóre je nad rozhodovacím pásmom, čo zodpovedá zvýšenému odhadu rizika "
            "pozitívneho výsledku HUTT testu. Výsledok je určený na posúdenie v kontexte "
            "klinického obrazu pacienta."
        )
    if label == "hraničné":
        return (
            "Modelové skóre je v okolí rozhodovacieho prahu. Ide o neisté pásmo, kde malá zmena "
            "vstupných údajov môže zmeniť klasifikáciu. Výsledok v tomto pásme má obmedzenú "
            "výpovednú hodnotu a mal by byť posudzovaný spolu s ďalšími dostupnými klinickými "
            "informáciami."
        )
    return (
        "Modelové skóre je pod rozhodovacím pásmom, čo zodpovedá nižšiemu odhadu rizika "
        "pozitívneho výsledku HUTT testu. Nízke modelové skóre nevylučuje klinicky významnú "
        "príčinu synkopy."
    )


def tri_state_input(label, key):
    choice = st.segmented_control(
        label,
        options=["Neznáme", "Áno", "Nie"],
        default="Neznáme",
        key=key,
    )
    if choice == "Áno":
        return 1.0
    if choice == "Nie":
        return 0.0
    return np.nan


def make_anamnesis_vector(values):
    return np.array([[
        values["Pohlavie_enc"],
        values["Vek"],
        values["TK_sys"],
        values["TK_dia"],
        values["Pulz"],
    ]], dtype=float)


def make_combined_vector(ana_values, dot_values):
    row = {f: np.nan for f in pkg_kom["features"]}
    row.update(ana_values)
    for k, v in dot_values.items():
        if k in row:
            row[k] = v
    return np.array([[row[f] for f in pkg_kom["features"]]], dtype=float)


def reset_case():
    for k in list(st.session_state.keys()):
        if k.startswith("case_") or k.startswith("dot_") or k == "_last_step":
            st.session_state.pop(k, None)
    st.session_state["case_step"] = "1_anamnesis"


def report_text():
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    ana = st.session_state.get("case_ana_values", {})
    prob_ana = st.session_state.get("case_prob_ana")
    prob_kom = st.session_state.get("case_prob_kom")
    dot_values = st.session_state.get("case_dot_values", {})

    lines = [
        "HUTT Decision Support – výskumný prototyp",
        f"Vygenerované: {now}",
        "",
        "UPOZORNENIE: Výstup predstavuje výskumný odhad a nie je klinicky validovaný diagnostický nástroj.",
        "",
        "ZÁKLADNÉ ÚDAJE",
        f"Pohlavie: {'Muž' if ana.get('Pohlavie_enc') == 1 else 'Žena'}",
        f"Vek: {ana.get('Vek')}",
        f"TK systolický: {ana.get('TK_sys')}",
        f"TK diastolický: {ana.get('TK_dia')}",
        f"Pulz: {ana.get('Pulz')}",
        "",
        "VÝSLEDOK",
    ]
    if prob_ana is not None:
        lines.append(f"Anamnéza ({pkg_ana.get('model_name')}): {prob_ana*100:.1f}% · prah {pkg_ana.get('threshold'):.2f}")
    if prob_kom is not None:
        lines.append(f"Kombinácia ({pkg_kom.get('model_name')}): {prob_kom*100:.1f}% · prah {pkg_kom.get('threshold'):.2f}")
        lines.append("")
        lines.append("Interpretácia:")
        lines.append(clinical_interpretation(prob_kom, pkg_kom["threshold"]))
    if dot_values:
        lines.append("")
        lines.append("DOTAZNÍKOVÉ PREMENNÉ")
        for k, v in dot_values.items():
            if k in NUMERIC_FEATURES:
                val = "Neznáme" if pd.isna(v) else str(int(v))
            else:
                val = "Áno" if v == 1 else ("Nie" if v == 0 else "Neznáme")
            lines.append(f"{k} – {FEATURE_LABELS.get(k, k)}: {val}")
    return "\n".join(lines)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🫀 HUTT Decision Support")
    st.caption("Výskumný prototyp · predikcia výsledku HUTT testu")
    st.divider()

    # ── Pracovný postup so zvýraznením aktuálneho kroku ──────────────────────
    _cur = st.session_state.get("case_step", "1_anamnesis")
    _done_ana = "case_prob_ana" in st.session_state
    _done_kom = "case_prob_kom" in st.session_state

    def _step_row(num, label, active, done):
        if done:
            icon, color, weight = "✅", "#1b8a5a", "600"
        elif active:
            icon, color, weight = "▶", "#0f5f8c", "700"
        else:
            icon, color, weight = "○", "#9aabbb", "400"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<span style='font-size:1rem;color:{color};'>{icon}</span>"
            f"<span style='color:{color};font-weight:{weight};font-size:0.93rem;'>{num}. {label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("**Pracovný postup**")
    _step_row("1", "Zadať klinické údaje",   active=(_cur == "1_anamnesis"),    done=_done_ana)
    _step_row("2", "Vyplniť dotazník",        active=(_cur == "2_questionnaire"), done=_done_kom)
    _step_row("3", "Interpretovať výsledok",  active=(_cur == "3_results"),       done=False)
    st.divider()

    # ── Modely ────────────────────────────────────────────────────────────────
    st.markdown("**Modely**")
    for _pkg, _label in [(pkg_ana, "Anamnéza"), (pkg_kom, "Kombinácia")]:
        _auc     = _pkg.get("AUC_CV", "?")
        _auc_std = _pkg.get("AUC_CV_std", "?")
        _thr     = _pkg.get("threshold")
        _thr_txt = f"{_thr:.2f}" if isinstance(_thr, float) else "?"
        st.markdown(
            f"<div style='margin-bottom:8px;'>"
            f"<span style='font-weight:700;font-size:0.9rem;'>{_label}</span><br>"
            f"<span style='color:#607080;font-size:0.82rem;'>"
            f"{_pkg.get('model_name','?')} &nbsp;·&nbsp; "
            f"AUC {_auc} ± {_auc_std} % &nbsp;·&nbsp; prah {_thr_txt}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
    st.caption(f"Dotazníkové premenné: {len(SELECTED_DOT)}")
    st.divider()

    # ── Upozornenie ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='warning-box' style='font-size:0.82rem;'>"
        "⚠️ <b>Výskumný prototyp.</b> Výstup nie je klinicky validovaný diagnostický nástroj "
        "a nenahrádza odborné posúdenie lekára."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if st.button("🔄 Nový pacient", use_container_width=True, key="sidebar_new_patient"):
        reset_case()
        st.session_state["_scroll_flag"] = True
        st.rerun()


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    """
<div class="hero">
    <h1>Predikcia výsledku HUTT testu</h1>
    <p>
    Výskumný model predikcie pozitívneho výsledku Head-Up Tilt Table Testu
    pri krátkodobej strate vedomia. Výstup predstavuje výskumný odhad modelového skóre
    a nie je náhradou klinického posúdenia.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if "case_step" not in st.session_state:
    st.session_state["case_step"] = "1_anamnesis"

step = st.session_state["case_step"]

_SCROLL_JS = """
<script>
(function() {
    var p = window.parent;
    try { p.document.documentElement.scrollTop = 0; } catch(e) {}
    try { p.document.body.scrollTop = 0; } catch(e) {}
    var sels = [
        'section[data-testid="stMain"] > div',
        'section[data-testid="stMain"]',
        '[data-testid="stAppViewContainer"] > section > div',
        '.main > div', '.main'
    ];
    for (var i = 0; i < sels.length; i++) {
        var el = p.document.querySelector(sels[i]);
        if (el) { el.scrollTop = 0; }
    }
})();
</script>
"""

_do_scroll = False
if st.session_state.get("_last_step") != step:
    st.session_state["_last_step"] = step
    _do_scroll = True
if st.session_state.pop("_scroll_flag", False):
    _do_scroll = True
if _do_scroll:
    components.html(_SCROLL_JS, height=1)

nav1, nav2, nav3, nav_new = st.columns([1, 1, 1, 0.75])
with nav1:
    if st.button("1. Anamnéza", use_container_width=True, type="primary" if step == "1_anamnesis" else "secondary"):
        st.session_state["case_step"] = "1_anamnesis"
        st.session_state["_scroll_flag"] = True
        st.rerun()
with nav2:
    disabled = "case_prob_ana" not in st.session_state
    if st.button("2. Dotazník", use_container_width=True, disabled=disabled, type="primary" if step == "2_questionnaire" else "secondary"):
        st.session_state["case_step"] = "2_questionnaire"
        st.session_state["_scroll_flag"] = True
        st.rerun()
with nav3:
    disabled = "case_prob_kom" not in st.session_state
    if st.button("3. Výsledok", use_container_width=True, disabled=disabled, type="primary" if step == "3_results" else "secondary"):
        st.session_state["case_step"] = "3_results"
        st.session_state["_scroll_flag"] = True
        st.rerun()
with nav_new:
    if st.button("🔄 Nový pacient", use_container_width=True, key="nav_new_patient"):
        reset_case()
        st.session_state["_scroll_flag"] = True
        st.rerun()

st.divider()


# =============================================================================
# STEP 1 — ANAMNESIS
# =============================================================================
if step == "1_anamnesis":
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### 1. Základné klinické údaje")
        st.caption("Údaje dostupné pred vyšetrením alebo v úvodnom kontakte s pacientom.")

        with st.form("anamnesis_form", border=False):
            c1, c2 = st.columns(2)
            with c1:
                sex = st.selectbox("Pohlavie", ["Žena", "Muž"])
                age = st.number_input("Vek pacienta", min_value=1, max_value=110, value=45, step=1)
            with c2:
                pulse_unknown = st.checkbox("Pulz nie je dostupný")
                pulse = np.nan if pulse_unknown else st.number_input("Pulz / min", min_value=20, max_value=220, value=70, step=1)

            st.markdown("**Tlak krvi (mmHg)**")
            bp_unknown = st.checkbox("Tlak krvi nie je dostupný")
            if bp_unknown:
                sys_bp, dia_bp = np.nan, np.nan
                st.caption("Chýbajúce hodnoty budú nahradené mediánom z trénovacej vzorky.")
            else:
                b1, b2 = st.columns(2)
                with b1:
                    sys_bp = st.number_input("Systolický TK (mmHg)", min_value=60, max_value=260, value=120, step=1)
                with b2:
                    dia_bp = st.number_input("Diastolický TK (mmHg)", min_value=30, max_value=160, value=80, step=1)

            submitted = st.form_submit_button("Vypočítať predbežné skóre", type="primary", use_container_width=True)

        if submitted:
            if not bp_unknown and sys_bp <= dia_bp:
                st.error("Systolický tlak musí byť vyšší než diastolický. Skontrolujte hodnoty.")
            else:
                ana_values = {
                    "Pohlavie_enc": 1.0 if sex == "Muž" else 0.0,
                    "Vek": float(age),
                    "TK_sys": float(sys_bp) if not pd.isna(sys_bp) else np.nan,
                    "TK_dia": float(dia_bp) if not pd.isna(dia_bp) else np.nan,
                    "Pulz": float(pulse) if not pd.isna(pulse) else np.nan,
                }
                x_ana = make_anamnesis_vector(ana_values)
                prob_ana = predict_proba(pkg_ana, x_ana)
                st.session_state["case_ana_values"] = ana_values
                st.session_state["case_prob_ana"] = prob_ana
                st.session_state["_scroll_flag"] = True
                st.rerun()

    with right:
        if "case_prob_ana" in st.session_state:
            score_card(
                title="Predbežné modelové skóre",
                prob=st.session_state["case_prob_ana"],
                threshold=float(pkg_ana["threshold"]),
                model_name=pkg_ana.get("model_name", "Anamnéza"),
                n_features=len(ANA_FEATURES),
                auc=pkg_ana.get("AUC_CV"),
                auc_std=pkg_ana.get("AUC_CV_std"),
            )
            st.markdown(
                f"<div class='notice'>{clinical_interpretation(st.session_state['case_prob_ana'], float(pkg_ana['threshold']))}</div>",
                unsafe_allow_html=True,
            )
            if st.button("Pokračovať na dotazník →", type="primary", use_container_width=True):
                st.session_state["case_step"] = "2_questionnaire"
                st.session_state["_scroll_flag"] = True
                st.rerun()
        else:
            st.markdown(
                """
<div class="soft-card">
<b>Čo aplikácia urobí?</b><br><br>
Najprv vypočíta predbežné skóre z piatich základných údajov. Potom je možné doplniť dotazníkové premenné a získať kombinované skóre.
</div>
<div class="warning-box">
<b>Dôležité:</b> Výstup nie je klinicky validovaná pravdepodobnosť. Ide o modelové skóre odvodené z poskytnutých retrospektívnych dát.
</div>
""",
                unsafe_allow_html=True,
            )


# =============================================================================
# STEP 2 — QUESTIONNAIRE
# =============================================================================
elif step == "2_questionnaire":
    if "case_ana_values" not in st.session_state:
        st.warning("Najprv vypočítajte predbežné skóre z anamnézy.")
        st.stop()

    st.markdown("### 2. Dotazníkové premenné")
    st.caption("Zvoľte Áno/Nie iba vtedy, ak je informácia dostupná. Pri neistote ponechajte Neznáme.")

    st.markdown(
        """
<div class="notice">
<b>Návod na vyplnenie:</b> Ak symptóm nebol prítomný, zvoľte <b>Nie</b>. Ak informácia nie je k dispozícii, zvoľte <b>Neznáme</b>.
</div>
""",
        unsafe_allow_html=True,
    )

    dot_values = {}
    shown = set()

    with st.form("questionnaire_form", border=False):
        for block_name, feature_list in BLOCK_ORDER:
            features_in_block = [f for f in feature_list if f in SELECTED_DOT]
            if not features_in_block:
                continue
            st.markdown(f"#### {block_name}")
            cols = st.columns(2)
            for i, feat in enumerate(features_in_block):
                shown.add(feat)
                with cols[i % 2]:
                    label = FEATURE_LABELS.get(feat, f"{feat} – doplňte text otázky")
                    if feat in NUMERIC_FEATURES:
                        unknown = st.checkbox(f"{label} — neznáme", key=f"dot_{feat}_unknown")
                        if unknown:
                            dot_values[feat] = np.nan
                        else:
                            _num_cfg = {"C1": (1, 100, 1), "C2": (0, 200, 0), "C4": (1, 100, 1)}
                            _min, _max, _def = _num_cfg.get(feat, (0, 200, 0))
                            dot_values[feat] = float(st.number_input(label, min_value=_min, max_value=_max, value=_def, step=1, key=f"dot_{feat}"))
                    else:
                        dot_values[feat] = tri_state_input(label, key=f"dot_{feat}")
            st.divider()

        # fallback for unexpected selected features not in block order
        rest = [f for f in SELECTED_DOT if f not in shown]
        if rest:
            st.markdown("#### Ostatné vybrané premenné")
            cols = st.columns(2)
            for i, feat in enumerate(rest):
                with cols[i % 2]:
                    label = FEATURE_LABELS.get(feat, f"{feat} – dotazníková premenná")
                    if feat in NUMERIC_FEATURES:
                        unknown = st.checkbox(f"{label} — neznáme", key=f"dot_{feat}_unknown")
                        dot_values[feat] = np.nan if unknown else float(st.number_input(label, value=0, step=1, key=f"dot_{feat}"))
                    else:
                        dot_values[feat] = tri_state_input(label, key=f"dot_{feat}")

        submitted = st.form_submit_button("Vypočítať kombinované skóre", type="primary", use_container_width=True)

    if submitted:
        x_kom = make_combined_vector(st.session_state["case_ana_values"], dot_values)
        prob_kom = predict_proba(pkg_kom, x_kom)
        st.session_state["case_dot_values"] = dot_values
        st.session_state["case_prob_kom"] = prob_kom
        st.session_state["case_step"] = "3_results"
        st.rerun()


# =============================================================================
# STEP 3 — RESULTS
# =============================================================================
elif step == "3_results":
    if "case_prob_kom" not in st.session_state:
        st.info("Výsledok sa zobrazí po vyplnení dotazníka.")
        st.stop()

    prob_ana = st.session_state["case_prob_ana"]
    prob_kom = st.session_state["case_prob_kom"]

    st.markdown("### 3. Klinický výstup")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        score_card(
            title="Model 1 — anamnéza",
            prob=prob_ana,
            threshold=float(pkg_ana["threshold"]),
            model_name=pkg_ana.get("model_name", "Anamnéza"),
            n_features=len(ANA_FEATURES),
            auc=pkg_ana.get("AUC_CV"),
            auc_std=pkg_ana.get("AUC_CV_std"),
        )
    with col2:
        score_card(
            title="Model 2 — kombinácia",
            prob=prob_kom,
            threshold=float(pkg_kom["threshold"]),
            model_name=pkg_kom.get("model_name", "Kombinácia"),
            n_features=len(pkg_kom.get("features", [])),
            auc=pkg_kom.get("AUC_CV"),
            auc_std=pkg_kom.get("AUC_CV_std"),
        )

    label_ana, _, _ = band(prob_ana, float(pkg_ana["threshold"]))
    label_kom, color_key_kom, _ = band(prob_kom, float(pkg_kom["threshold"]))

    if label_ana == label_kom:
        st.success(f"Oba modely sa zhodujú: skóre je v pásme **{label_kom}**.")
    else:
        st.warning(
            f"Modely sa líšia: anamnéza je v pásme **{label_ana}**, kombinácia v pásme **{label_kom}**. "
            "Kombinovaný model zohľadňuje aj dotazníkové premenné. Pri rozdielnych výsledkoch "
            "je vhodné zvážiť oba pohľady v kontexte dostupných klinických informácií."
        )

    delta = prob_kom - prob_ana
    st.markdown(
        f"""
<div class="clinical-card">
    <div class="metric-title">Kontextová informácia k výsledku</div>
    <p>{clinical_interpretation(prob_kom, float(pkg_kom['threshold']))}</p>
    <div class="small-muted">
        Zmena po doplnení dotazníka: <b>{delta*100:+.1f} percentuálneho bodu</b>
        ({prob_ana*100:.1f}% → {prob_kom*100:.1f}%).
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    filled = 0
    dot_values = st.session_state.get("case_dot_values", {})
    for val in dot_values.values():
        if not pd.isna(val):
            filled += 1

    q1, q2, q3 = st.columns(3)
    q1.metric("Vyplnené dotazníkové premenné", f"{filled}/{len(SELECTED_DOT)}")
    q2.metric("Prahová hodnota kombinovaného modelu", f"{float(pkg_kom['threshold']):.2f}")
    q3.metric("Modelové skóre", f"{prob_kom*100:.1f}%")

    # ── Distribučné grafy ────────────────────────────────────────────────────
    def _draw_hist(ax, pkg, prob_patient, title):
        _pos = pkg.get("train_proba_pos", [])
        _neg = pkg.get("train_proba_neg", [])
        if not _pos or not _neg:
            return False
        _bins = np.linspace(0, 1, 21)
        ax.hist(_neg, bins=_bins, alpha=0.65, color="#1b8a5a", label=f"HUTT− (n={len(_neg)})",
                density=True, edgecolor="white", linewidth=0.5)
        ax.hist(_pos, bins=_bins, alpha=0.65, color="#b83b3b", label=f"HUTT+ (n={len(_pos)})",
                density=True, edgecolor="white", linewidth=0.5)
        ax.axvline(prob_patient, color="#c97918", linewidth=2, linestyle="--",
                   label=f"Pacient ({round(prob_patient*100,1)}%)")
        ax.axvline(pkg["threshold"], color="#0f5f8c", linewidth=1.5, linestyle=":",
                   label=f"Prah ({int(pkg['threshold']*100)}%)")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Modelové skóre", fontsize=8)
        ax.set_ylabel("Hustota", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper center", framealpha=0.85)
        ax.set_xlim(0, 1)
        ax.spines[['top', 'right']].set_visible(False)
        return True

    with st.expander("📊 Distribúcia modelového skóre", expanded=False):
        st.caption("Poloha pacienta voči HUTT− a HUTT+ pacientom z trénovacej vzorky")
        _sp1, _gc1, _gc2, _sp2 = st.columns([0.3, 2, 2, 0.3])
        with _gc1:
            _fig1, _ax1 = plt.subplots(figsize=(3.6, 2.6), dpi=110)
            if _draw_hist(_ax1, pkg_ana, prob_ana,
                          f"Anamnéza — {pkg_ana.get('model_name', 'ExtraTrees')}"):
                _fig1.tight_layout()
                st.pyplot(_fig1, use_container_width=False)
            else:
                st.info("Distribučné dáta nie sú dostupné.")
            plt.close(_fig1)
        with _gc2:
            _fig2, _ax2 = plt.subplots(figsize=(3.6, 2.6), dpi=110)
            if _draw_hist(_ax2, pkg_kom, prob_kom,
                          f"Kombinácia — {pkg_kom.get('model_name', 'RF')}"):
                _fig2.tight_layout()
                st.pyplot(_fig2, use_container_width=False)
            else:
                st.info("Distribučné dáta nie sú dostupné.")
            plt.close(_fig2)

    st.markdown(
        """
<div class="warning-box">
Model nebol externe prospektívne validovaný. Výstup predstavuje výskumný odhad a nie je určený ako samostatný podklad klinického rozhodnutia. Ide o doplnkovú informáciu k odbornému posúdeniu.
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Zobraziť zadané údaje pacienta"):
        ana = st.session_state.get("case_ana_values", {})
        ana_df = pd.DataFrame([
            {"Premenná": "Pohlavie", "Hodnota": "Muž" if ana.get("Pohlavie_enc") == 1 else "Žena"},
            {"Premenná": "Vek", "Hodnota": ana.get("Vek")},
            {"Premenná": "TK systolický", "Hodnota": ana.get("TK_sys")},
            {"Premenná": "TK diastolický", "Hodnota": ana.get("TK_dia")},
            {"Premenná": "Pulz", "Hodnota": ana.get("Pulz")},
        ])
        st.dataframe(ana_df, use_container_width=True, hide_index=True)

        dot_df = []
        for k, v in dot_values.items():
            if k in NUMERIC_FEATURES:
                value = "Neznáme" if pd.isna(v) else int(v)
            else:
                value = "Áno" if v == 1 else ("Nie" if v == 0 else "Neznáme")
            dot_df.append({"Kód": k, "Premenná": FEATURE_LABELS.get(k, k), "Hodnota": value})
        if dot_df:
            st.dataframe(pd.DataFrame(dot_df), use_container_width=True, hide_index=True)

    with st.expander("Model Card"):
        n_total = pkg_kom.get("n_train", "?")
        if isinstance(n_total, int) and isinstance(pkg_kom.get("n_test"), int):
            n_total = pkg_kom.get("n_train") + pkg_kom.get("n_test")
        st.markdown(f"""
**Účel:** Orientačný odhad modelového skóre pre pozitívny HUTT test.  
**Cieľová premenná:** A10 – výsledok HUTT testu.  
**Anamnestický model:** {pkg_ana.get('model_name')} · prah {float(pkg_ana['threshold']):.2f}.  
**Kombinovaný model:** {pkg_kom.get('model_name')} · prah {float(pkg_kom['threshold']):.2f}.  
**Počet vstupných premenných v kombinovanom modeli:** {len(pkg_kom.get('features', []))}.  
**Validácia:** interná validácia na poskytnutom klinickom datasete; externá validácia chýba.  
**Limitácie:** retrospektívne dáta, malá vzorka, možné chýbajúce alebo neúplné odpovede, jednocentrický pôvod dát.  
""")

    st.download_button(
        "⬇️ Stiahnuť textový report",
        data=report_text().encode("utf-8"),
        file_name=f"hutt_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


st.divider()
st.caption("Bakalárska práca · HUTT predikcia · Výskumný prototyp · Streamlit")
