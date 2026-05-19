# -*- coding: utf-8 -*-
"""
app.py — HUTT Decision Support
Streamlit aplikácia pre predikciu výsledku HUTT testu.
Modely: analyza.py (ConsensusFS prahová logika, P1 ET + P3 ET)

Spustenie:
    streamlit run app.py

Vyžaduje v priečinku modely_aplikacia/:
    model_p1_et.joblib
    model_p3_et.joblib
"""

import math
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from model_components import ConsensusFeatureSelector, P3SelectorConsensus


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
# CSS
# =============================================================================
st.markdown("""
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
    --yellow: #b8860b;
    --orange: #c97918;
    --red: #b83b3b;
    --shadow: 0 8px 24px rgba(15,35,60,0.08);
}
.block-container { padding-top: 1.7rem; }
.hero {
    background: linear-gradient(135deg, #0f5f8c 0%, #143a5a 100%);
    color: white; padding: 28px 32px; border-radius: 22px;
    box-shadow: var(--shadow); margin-bottom: 20px;
}
.hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: -0.02em; }
.hero p  { margin-top: 10px; color: rgba(255,255,255,0.88); font-size: 1rem; max-width: 980px; }
.clinical-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 18px; padding: 22px 24px;
    box-shadow: var(--shadow); margin-bottom: 16px;
}
.soft-card {
    background: var(--bg-soft); border: 1px solid var(--border);
    border-radius: 16px; padding: 16px 18px; margin-bottom: 12px;
}
.metric-title { color: var(--muted); font-size: 0.86rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .05em; }
.metric-big { font-size: 3.4rem; line-height: 1; font-weight: 800; margin: 7px 0; }
.metric-sub { color: var(--muted); font-size: 0.88rem; }
.badge { display: inline-block; padding: 6px 10px; border-radius: 999px;
    font-weight: 700; font-size: 0.86rem; margin-top: 8px; }
.badge-green  { background: rgba(27,138,90,.12);  color: var(--green); }
.badge-yellow { background: #fff3cd; color: #7d5a00; border: 1px solid #e6c85a; }
.badge-orange { background: rgba(201,121,24,.14); color: var(--orange); }
.badge-red    { background: rgba(184,59,59,.13);  color: var(--red); }
.progress-track { background: #e8eef5; border-radius: 999px; height: 8px;
    margin: 8px 0; overflow: hidden; }
.progress-fill  { height: 100%; border-radius: 999px; transition: width .4s; }
.notice { border-left: 4px solid var(--primary); background: #eef4fa;
    padding: 13px 16px; border-radius: 12px; color: #1a3a5a; margin: 12px 0; }
.warning-box { border-left: 5px solid var(--red); background: #fff0f0;
    padding: 13px 16px; border-radius: 12px; color: #3a1717; margin: 12px 0; }
.sens-spec-row { font-size: 0.83rem; color: var(--muted); margin-top: 6px; }
button[kind="primary"], .stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
    color: white !important; border-radius: 12px !important;
    min-height: 42px; font-weight: 700 !important;
}
button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background-color: var(--primary-dark) !important;
    border-color: var(--primary-dark) !important;
}
.stButton > button { border-radius: 12px !important; min-height: 40px; }
.small-muted { color: var(--muted); font-size: .86rem; }
hr { margin: 1.1rem 0; }
[data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING
# =============================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PATH_P1   = os.path.join(BASE_DIR, "model_p1_et.joblib")
PATH_P3   = os.path.join(BASE_DIR, "model_p3_et.joblib")


@st.cache_resource(show_spinner=False)
def load_models():
    p1 = joblib.load(PATH_P1)
    p3 = joblib.load(PATH_P3)
    for pkg, name, required in [
        (p1, "P1", {"pipeline", "features", "threshold", "model_name"}),
        (p3, "P3", {"pipeline", "features", "threshold", "model_name",
                    "p2_selected_features"}),
    ]:
        missing = required - set(pkg.keys())
        if missing:
            raise ValueError(f"{name} model chýba kľúče: {missing}")
    return p1, p3


try:
    pkg_p1, pkg_p3 = load_models()
except Exception as e:
    st.error("Modelové súbory sa nepodarilo načítať.")
    st.exception(e)
    st.stop()

THRESHOLD_P1   = 0.5
THRESHOLD_P3   = 0.5
P1_FEATURES    = pkg_p1["features"]
P3_ALL_FEATS   = pkg_p3["features"]        # celý P3_POOL (vstup pipeline)
P3_DOT_FEATS   = pkg_p3["p2_selected_features"]  # len dotazníkové (FS vybrané)


# =============================================================================
# FEATURE LABELS
# =============================================================================
FORM_TO_INTERNAL = {
    "TK_sys":       "A2_sys",
    "TK_dia":       "A2_dia",
    "Pulz":         "A3",
    "Pohlavie_enc": "Pohlavie",
    "Vek":          "Vek",
}

FEATURE_LABELS = {
    "Pohlavie": "Pohlavie", "Pohlavie_enc": "Pohlavie",
    "Vek": "Vek", "TK_sys": "Systolický tlak krvi",
    "TK_dia": "Diastolický tlak krvi", "Pulz": "Pulz",
    "A2_sys": "Systolický tlak krvi", "A2_dia": "Diastolický tlak krvi",
    "A3": "Pulz (tepová frekvencia)",
    "C1": "Vek pri prvom výskyte ťažkostí",
    "C2": "Celkový počet epizód straty vedomia",
    "C4": "Vek v období najhorších ťažkostí",
    "D1": "Strata vedomia v sede",
    "D2": "Strata vedomia do 1 min po postavení sa",
    "D3": "Strata vedomia pri chôdzi",
    "D4": "Strata vedomia v ľahu",
    "D5": "Strata vedomia pri fyzickej námahe",
    "D6": "Strata vedomia po fyzickej námahe",
    "E1": "Strata vedomia v preľudnených priestoroch",
    "E2": "Strata vedomia pri dlhom státí",
    "E3": "Strata vedomia pri teplote/horúčave",
    "E4": "Strata vedomia po pohľade na krv",
    "E5": "Strata vedomia po nepríjemných emóciách",
    "E6": "Strata vedomia po intenzívnom kašli",
    "E7": "Strata vedomia po bolesti",
    "E8": "Strata vedomia pri dehydratácii",
    "E9": "Strata vedomia pri menštruácii",
    "F1": "Strata vedomia pri stolici",
    "F2": "Strata vedomia pri močení",
    "F3": "Strata vedomia pri kašli/smrkaní",
    "F4": "Strata vedomia pri kýchaní",
    "F5": "Strata vedomia po jedle",
    "F6": "Strata vedomia po náhlej bolesti",
    "F7": "Strata vedomia počas fyzickej námahy",
    "F10": "Strata vedomia z iného dôvodu",
    "H1": "Nevoľnosť/vracanie pred stratou vedomia",
    "H2": "Pocit tepla/horúčavy pred stratou vedomia",
    "H3": "Potenie pred stratou vedomia",
    "H4": "Zahmlievanie pred očami",
    "H5": "Hučanie v ušiach pred stratou vedomia",
    "H6": "Slabosť pred stratou vedomia",
    "H10": "Neobvyklé zvuky pred stratou vedomia",
    "H11": "Búšenie srdca pred stratou vedomia",
    "H12": "Bolesť na hrudníku pred stratou vedomia",
    "H13": "Pacient si nepamätá pocity pred stratou vedomia",
    "H14": "Iné príznaky pred stratou vedomia",
    "I1": "Príznaky trvali niekoľko sekúnd",
    "I2": "Príznaky trvali do 1 minúty",
    "I3": "Príznaky trvali viac ako 1 minútu",
    "I4": "Príznaky trvali rôzne dlho",
    "J3": "Pulz počas epizódy (meranie svedkom)",
    "K1": "Bezvedomie trvalo niekoľko sekúnd",
    "K2": "Bezvedomie trvalo do 1 minúty",
    "K3": "Bezvedomie trvalo 1–5 minút",
    "K4": "Bezvedomie trvalo viac ako 5 minút",
    "L":  "Kŕče počas bezvedomia",
    "M":  "Inkontinencia počas bezvedomia",
    "N1": "Pohryzený jazyk/pery po strate vedomia",
    "N3": "Zmätenosť po prebratí",
    "N4": "Bolesť hlavy/svalov po prebratí",
    "N5": "Únava po prebratí",
    "N6": "Pacient sa cítil normálne po prebratí",
    "O1": "Náhle úmrtie člena rodiny",
    "O2": "Náhle srdcové zlyhanie v rodine",
    "O3": "Iné ochorenie srdca v rodine",
    "O4": "Synkopa v rodine",
    "O5": "Iné kardiovaskulárne ochorenia v rodine",
    "P1": "Chlopňové ochorenie srdca",
    "P2": "Kardiomyopatia",
    "P3": "Vrodené ochorenie srdca",
    "P4": "Arytmia",
    "P5": "Koronárna choroba srdca",
    "P6": "Srdcové zlyhávanie",
    "P7": "Iné ochorenie srdca",
    "P9": "Bolesti na hrudníku",
    "P10": "Dýchavičnosť",
    "P11": "Nízky krvný tlak",
    "P12": "Závraty",
    "P13": "Búšenie srdca",
    "P15": "Vysoký krvný tlak",
    "P16": "Astma",
    "P17": "Cukrovka",
    "P18": "Ochorenia priedušiek",
    "P21": "Epilepsia",
    "P23": "Migréna",
    "P26": "Úzkosť alebo panická porucha",
    "P27": "Depresia",
    "P28": "Lieky na krvný tlak",
    "P29": "Antidepresíva",
    "P31": "Nádorové ochorenie",
    "P33": "Prekonané úrazy",
    "P34": "Iné ochorenia",
    "Q3": "Antiarytmiká",
    "Q5": "Diuretiká",
    "Q6": "Betablokátory",
    "Q8": "Nitroglycerín",
    "Q10": "Antikoagulanciá",
    "Q11": "Antidiabetiká",
    "Q15": "Iné lieky",
    "R1": "Predchádzajúce EKG",
    "R2": "Predchádzajúce Holter EKG",
    "Ma_diag_srdcove_ochorenie": "Diagnostikované srdcové ochorenie",
}

NUMERIC_FEATURES = {"C1", "C2", "C4", "J3"}

# Blokové usporiadanie dotazníka (pre Step 2)
BLOCK_ORDER = [
    ("Vznik a charakter ťažkostí",    ["C1", "C2", "C4"]),
    ("Okolnosti straty vedomia",       ["D1", "D2", "D3", "D4", "D5", "D6"]),
    ("Spúšťacie faktory straty vedomia", ["E1", "E2", "E3", "E4", "E5", "E6",
                                        "E7", "E8", "E9", "F1", "F2", "F3",
                                        "F4", "F5", "F6", "F7", "F10"]),
    ("Príznaky pred stratou vedomia", ["H1", "H2", "H3", "H4", "H5", "H6",
                                        "H10", "H11", "H12", "H13", "H14",
                                        "I1", "I2", "I3", "I4"]),
    ("Priebeh a stav po udalosti",    ["J3", "K1", "K2", "K3", "K4",
                                        "L", "M", "N1", "N3", "N4", "N5", "N6"]),
    ("Rodinná anamnéza",              ["O1", "O2", "O3", "O4", "O5"]),
    ("Osobná anamnéza",               ["P1", "P2", "P3", "P4", "P5", "P6", "P7",
                                        "P9", "P10", "P11", "P12", "P13", "P15",
                                        "P16", "P17", "P18", "P21", "P23",
                                        "P26", "P27", "P28", "P29", "P31",
                                        "P33", "P34"]),
    ("Lieky a vyšetrenia",            ["Q3", "Q5", "Q6", "Q8", "Q10", "Q11",
                                        "Q15", "R1", "R2"]),
]


# =============================================================================
# HELPERS
# =============================================================================
def band(prob: float):
    if prob >= 0.75:
        return "vysoké", "red",    "Výraznejšia pravdepodobnosť pozitívneho výsledku"
    if prob >= 0.60:
        return "zvýšené", "orange", "Zvýšená pravdepodobnosť pozitívneho výsledku"
    if prob >= 0.40:
        return "hraničné", "yellow", "Hraničná / neistá zóna"
    return "nízke", "green", "Skôr negatívny výsledok"


def band_color(color_key):
    return {
        "green":  "#1b8a5a",
        "yellow": "#b8860b",
        "orange": "#c97918",
        "red":    "#b83b3b",
    }[color_key]


def badge_class(color_key):
    return {
        "green":  "badge-green",
        "yellow": "badge-yellow",
        "orange": "badge-orange",
        "red":    "badge-red",
    }[color_key]




def score_card(title, prob, pkg):
    label, color_key, headline = band(prob)
    color  = band_color(color_key)
    pct    = round(prob * 100, 1)
    thr    = float(pkg.get("threshold", 0.5))
    auc    = pkg.get("AUC_CV_mean")
    std    = pkg.get("AUC_std")
    sens   = pkg.get("sensitivity_at_thr")
    spec   = pkg.get("specificity_at_thr")
    ppv    = pkg.get("ppv_at_thr")
    npv    = pkg.get("npv_at_thr")
    n_sel  = len(pkg.get("selected_features", pkg.get("features", [])))

    auc_str  = f"AUC {round(auc*100,1)}%" if auc else "AUC –"
    std_str  = f" ±{round(std*100,1)}%" if std else ""
    sens_str = (f"Sens {round(sens*100,0):.0f}% &nbsp;·&nbsp; "
                f"Spec {round(spec*100,0):.0f}% &nbsp;·&nbsp; "
                f"PPV {round(ppv*100,0):.0f}% &nbsp;·&nbsp; "
                f"NPV {round(npv*100,0):.0f}%"
                if sens and spec and ppv and npv else "")

    st.markdown(f"""
<div class="clinical-card">
    <div class="metric-title">{title}</div>
    <div class="metric-big" style="color:{color};">{pct}%</div>
    <div class="progress-track">
        <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <span class="badge {badge_class(color_key)}">{headline}</span>
    <div class="metric-sub" style="margin-top:10px;">
        Model: <b>{pkg.get('model_name','?')}</b> &nbsp;·&nbsp;
        Premenné: <b>{n_sel}</b> &nbsp;·&nbsp;
        {auc_str}{std_str}
    </div>
    <div class="sens-spec-row">
        Pri prahu <b>{thr:.2f}</b>: &nbsp; {sens_str}
    </div>
</div>
""", unsafe_allow_html=True)


def clinical_interpretation(prob: float) -> str:
    if prob >= 0.75:
        return (
            "Modelové skóre je v pásme výraznejšej pravdepodobnosti pozitívneho výsledku "
            "HUTT testu. Výsledok je určený na posúdenie v kontexte klinického obrazu "
            "pacienta — nie je náhradou odborného vyšetrenia."
        )
    if prob >= 0.60:
        return (
            "Modelové skóre naznačuje zvýšenú pravdepodobnosť pozitívneho výsledku "
            "HUTT testu. Výsledok je určený na posúdenie v kontexte klinického obrazu "
            "pacienta — nie je náhradou odborného vyšetrenia."
        )
    if prob >= 0.40:
        return (
            "Modelové skóre je v hraničnom pásme neistoty. Malá zmena vstupných údajov "
            "môže zmeniť zaradenie. Výsledok má obmedzenú výpovednú hodnotu a mal by byť "
            "posudzovaný spolu s ďalšími klinickými informáciami."
        )
    return (
        "Modelové skóre naznačuje skôr negatívny výsledok HUTT testu. "
        "Nízke modelové skóre nevylučuje klinicky významnú príčinu synkopy."
    )


def tri_state_input(label, key):
    choice = st.segmented_control(
        label, options=["Neznáme", "Áno", "Nie"],
        default="Neznáme", key=key,
    )
    return 1.0 if choice == "Áno" else (0.0 if choice == "Nie" else np.nan)


def make_p1_vector(form_values):
    row = {}
    for form_k, val in form_values.items():
        internal = FORM_TO_INTERNAL.get(form_k, form_k)
        row[internal] = val
    return pd.DataFrame(
        [[row.get(f, np.nan) for f in P1_FEATURES]], columns=P1_FEATURES
    )


def make_p3_vector(form_values, dot_values):
    """Zostrojí vstupný vektor pre celý P3_POOL (NaN pre nevyplnené features)."""
    row = {f: np.nan for f in P3_ALL_FEATS}
    for form_k, val in form_values.items():
        internal = FORM_TO_INTERNAL.get(form_k, form_k)
        if internal in row:
            row[internal] = val
    for k, v in dot_values.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([[row[f] for f in P3_ALL_FEATS]], columns=P3_ALL_FEATS)


def predict_p1(X_p1):
    return float(pkg_p1["pipeline"].predict_proba(X_p1)[0, 1])


def predict_p3(X_p3):
    return float(pkg_p3["pipeline"].predict_proba(X_p3)[0, 1])


def reset_case():
    for k in list(st.session_state.keys()):
        if k.startswith("case_") or k.startswith("dot_") or k == "_last_step":
            st.session_state.pop(k, None)
    st.session_state["case_step"] = "1_anamnesis"





# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🫀 HUTT Decision Support")
    st.caption("Výskumný prototyp · predikcia výsledku HUTT testu")
    st.divider()

    _cur     = st.session_state.get("case_step", "1_anamnesis")
    _done_p1 = "case_prob_p1" in st.session_state
    _done_p3 = "case_prob_p3" in st.session_state

    def _step_row(num, label, active, done):
        icon, color, weight = (
            ("✅", "#1b8a5a", "600") if done else
            ("▶",  "#0f5f8c", "700") if active else
            ("○",  "#9aabbb", "400")
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<span style='font-size:1rem;color:{color};'>{icon}</span>"
            f"<span style='color:{color};font-weight:{weight};"
            f"font-size:0.93rem;'>{num}. {label}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("**Pracovný postup**")
    _step_row("1", "Zadať klinické údaje",  active=(_cur == "1_anamnesis"),     done=_done_p1)
    _step_row("2", "Vyplniť dotazník",       active=(_cur == "2_questionnaire"), done=_done_p3)
    _step_row("3", "Interpretovať výsledok", active=(_cur == "3_results"),       done=False)
    st.divider()

    st.markdown("**Modely**")
    for _pkg, _label in [
        (pkg_p1, "P1 — anamnéza (ET, 5 premenných)"),
        (pkg_p3, f"P3 — kombinácia (ET, "
                 f"{len(pkg_p3.get('selected_features', []))} premenných)"),
    ]:
        _auc = _pkg.get("AUC_CV_mean")
        _std = _pkg.get("AUC_std")
        _thr = _pkg.get("threshold")
        _sens = _pkg.get("sensitivity_at_thr", 0)
        _spec = _pkg.get("specificity_at_thr", 0)
        st.markdown(
            f"<div style='margin-bottom:8px;'>"
            f"<span style='font-weight:700;font-size:0.9rem;'>{_label}</span><br>"
            f"<span style='color:#607080;font-size:0.82rem;'>"
            f"AUC {round(_auc*100,1) if _auc else '–'}%"
            f"{f' ±{round(_std*100,1)}%' if _std else ''}"
            f" &nbsp;·&nbsp; prah {f'{_thr:.2f}' if isinstance(_thr, float) else '–'}"
            f"<br>Sens {round(_sens*100,0):.0f}% / Spec {round(_spec*100,0):.0f}%"
            f"</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()

    st.markdown("**Farebné zóny skóre**")
    st.markdown("""
<div style="font-size:0.82rem; line-height:1.7;">
<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#1b8a5a;margin-right:6px;vertical-align:middle;"></span><b style="color:#1b8a5a;">< 40 %</b> — skôr negatívny<br>
<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#b8860b;margin-right:6px;vertical-align:middle;"></span><b style="color:#b8860b;">40 – 60 %</b> — hraničná zóna<br>
<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#c97918;margin-right:6px;vertical-align:middle;"></span><b style="color:#c97918;">60 – 75 %</b> — zvýšená pravdepodobnosť<br>
<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#b83b3b;margin-right:6px;vertical-align:middle;"></span><b style="color:#b83b3b;">≥ 75 %</b> — výraznejšia pravdepodobnosť
</div>
""", unsafe_allow_html=True)
    st.divider()

    st.markdown(
        "<div class='warning-box' style='font-size:0.82rem;'>"
        "⚠️ <b>Výskumný prototyp.</b> Výstup nie je klinicky validovaný diagnostický "
        "nástroj a nenahrádza odborné posúdenie lekára."
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Nový pacient", use_container_width=True, key="sidebar_new_patient"):
        reset_case()
        st.session_state["_scroll_flag"] = True
        st.rerun()


# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="hero">
    <h1>Predikcia výsledku HUTT testu</h1>
    <p>
    Výskumný model predikcie pozitívneho výsledku Head-Up Tilt Table Testu
    pri krátkodobej strate vedomia. Porovnanie modelu P1 (anamnéza, 5 premenných)
    a P3 (kombinácia anamnézy a dotazníka). Výstup predstavuje výskumný odhad
    a nie je náhradou klinického posúdenia.
    </p>
</div>
""", unsafe_allow_html=True)

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
    if st.button("1. Anamnéza", use_container_width=True,
                 type="primary" if step == "1_anamnesis" else "secondary"):
        st.session_state["case_step"] = "1_anamnesis"
        st.session_state["_scroll_flag"] = True
        st.rerun()
with nav2:
    if st.button("2. Dotazník", use_container_width=True,
                 disabled="case_prob_p1" not in st.session_state,
                 type="primary" if step == "2_questionnaire" else "secondary"):
        st.session_state["case_step"] = "2_questionnaire"
        st.session_state["_scroll_flag"] = True
        st.rerun()
with nav3:
    if st.button("3. Výsledok", use_container_width=True,
                 disabled="case_prob_p3" not in st.session_state,
                 type="primary" if step == "3_results" else "secondary"):
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
# STEP 1 — ANAMNESIS  →  P1 ET
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
                age = st.number_input("Vek pacienta", min_value=1,
                                      max_value=110, value=45, step=1)
            with c2:
                pulse_unknown = st.checkbox("Pulz nie je dostupný")
                pulse = np.nan if pulse_unknown else st.number_input(
                    "Pulz / min", min_value=20, max_value=220, value=70, step=1)

            st.markdown("**Tlak krvi (mmHg)**")
            bp_unknown = st.checkbox("Tlak krvi nie je dostupný")
            if bp_unknown:
                sys_bp, dia_bp = np.nan, np.nan
                st.caption("Chýbajúce hodnoty budú nahradené mediánom z trénovacej vzorky.")
            else:
                b1, b2 = st.columns(2)
                with b1:
                    sys_bp = st.number_input("Systolický TK (mmHg)",
                                             min_value=60, max_value=260,
                                             value=120, step=1)
                with b2:
                    dia_bp = st.number_input("Diastolický TK (mmHg)",
                                             min_value=30, max_value=160,
                                             value=80, step=1)

            submitted = st.form_submit_button("Vypočítať predbežné skóre (P1)",
                                              type="primary", use_container_width=True)

        if submitted:
            if not bp_unknown and sys_bp <= dia_bp:
                st.error("Systolický tlak musí byť vyšší než diastolický.")
            else:
                form_values = {
                    "Pohlavie_enc": 1.0 if sex == "Muž" else 0.0,
                    "Vek":    float(age),
                    "TK_sys": float(sys_bp) if not pd.isna(sys_bp) else np.nan,
                    "TK_dia": float(dia_bp) if not pd.isna(dia_bp) else np.nan,
                    "Pulz":   float(pulse)  if not pd.isna(pulse)  else np.nan,
                }
                X_p1    = make_p1_vector(form_values)
                prob_p1 = predict_p1(X_p1)
                st.session_state["case_ana_values"] = form_values
                st.session_state["case_X_p1"]       = X_p1
                st.session_state["case_prob_p1"]    = prob_p1
                st.session_state["_scroll_flag"]    = True
                st.rerun()

    with right:
        if "case_prob_p1" in st.session_state:
            score_card(
                title="Predbežné skóre — P1 (anamnéza)",
                prob=st.session_state["case_prob_p1"],
                pkg=pkg_p1,
            )
            st.markdown(
                f"<div class='notice'>"
                f"{clinical_interpretation(st.session_state['case_prob_p1'])}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Pokračovať na dotazník →", type="primary",
                         use_container_width=True):
                st.session_state["case_step"] = "2_questionnaire"
                st.session_state["_scroll_flag"] = True
                st.rerun()
        else:
            st.markdown("""
<div class="soft-card">
<b>Čo aplikácia urobí?</b><br><br>
Najprv vypočíta predbežné skóre z piatich základných údajov (pohlavie, vek, TK, pulz)
pomocou modelu <b>P1 Extra Trees</b> (AUC ≈ 80 %).<br><br>
Potom je možné doplniť dotazníkové premenné a získať kombinované skóre
modelu <b>P3 Extra Trees</b> (anamnéza + dotazník, AUC ≈ 81 %).
</div>
<div class="notice">
<b>Pre koho je aplikácia určená:</b><br>
Pre pacientov odoslaných na HUTT vyšetrenie s niektorou z týchto indikácií:<br>
strata vedomia · pocit hroziacej straty vedomia · opakované pády · stav po epileptickom záchvate · stav po resuscitácii.<br>
Skóre odráža riziko v kontexte tejto predselektovanej skupiny.<br>
Pacienti po resuscitácii (n=2) a po epileptickom záchvate (n=8) sú v tréningových dátach zastúpení minimálne, predikcie pre nich interpretujte s mimoriadnou opatrnosťou.
</div>

""".format(
                n_train=pkg_p3.get("n_train", "?"),
                n_pos=pkg_p3.get("n_pos", "?"),
                prev=round(pkg_p3.get("n_pos", 0) / pkg_p3.get("n_train", 1) * 100)
                     if pkg_p3.get("n_train") else 0,
            ), unsafe_allow_html=True)


# =============================================================================
# STEP 2 — QUESTIONNAIRE  →  P3 ET
# =============================================================================
elif step == "2_questionnaire":
    if "case_ana_values" not in st.session_state:
        st.warning("Najprv vypočítajte predbežné skóre z anamnézy.")
        st.stop()

    st.markdown("### 2. Dotazníkové premenné")
    st.caption(
        f"Model P3 používa {len(P3_DOT_FEATS)} dotazníkových premenných "
        f"vybraných ConsensusFS. Zvoľte Áno/Nie iba vtedy, ak je informácia dostupná."
    )
    st.markdown("""
<div class="notice">
<b>Dôležitý kontext:</b> Niektoré otázky sa týkajú <b>priebehu synkopálnej alebo presynkopálnej príhody</b>
(príznaky pred stratou vedomia, okolnosti, stav po prebudení).
Vypĺňajte ich len ak pacient takúto príhodu prekonal a okolnosti sú známe.
Ak informácia nie je dostupná, ponechajte pole na <b>Neznáme</b>.
</div>
""", unsafe_allow_html=True)

    dot_values = {}
    shown      = set()

    with st.form("questionnaire_form", border=False):
        for block_name, feature_list in BLOCK_ORDER:
            features_in_block = [f for f in feature_list if f in P3_DOT_FEATS]
            if not features_in_block:
                continue
            st.markdown(f"#### {block_name}")
            cols = st.columns(2)
            for i, feat in enumerate(features_in_block):
                shown.add(feat)
                with cols[i % 2]:
                    label = FEATURE_LABELS.get(feat, f"{feat}")
                    if feat in NUMERIC_FEATURES:
                        # C1/C4: ponúknuť aj možnosť "ťažkosti sa nevyskytli"
                        if feat in ("C1", "C4"):
                            _no_sx = st.checkbox(
                                f"{label} — ťažkosti sa nevyskytli / neznáme",
                                key=f"dot_{feat}_unknown")
                        else:
                            _no_sx = st.checkbox(f"{label} — neznáme",
                                                 key=f"dot_{feat}_unknown")
                        if _no_sx:
                            dot_values[feat] = np.nan
                        else:
                            _cfgs = {"C1": (1, 100, 1), "C2": (0, 200, 0),
                                     "C4": (1, 100, 1), "J3": (20, 220, 70)}
                            _min, _max, _def = _cfgs.get(feat, (0, 200, 0))
                            dot_values[feat] = float(st.number_input(
                                label, min_value=_min, max_value=_max,
                                value=_def, step=1, key=f"dot_{feat}"))
                    else:
                        dot_values[feat] = tri_state_input(label, key=f"dot_{feat}")
            st.divider()

        rest = [f for f in P3_DOT_FEATS if f not in shown]
        if rest:
            st.markdown("#### Ostatné vybrané premenné")
            cols = st.columns(2)
            for i, feat in enumerate(rest):
                with cols[i % 2]:
                    label = FEATURE_LABELS.get(feat, f"{feat}")
                    if feat in NUMERIC_FEATURES:
                        unknown = st.checkbox(f"{label} — neznáme",
                                              key=f"dot_{feat}_unknown")
                        dot_values[feat] = np.nan if unknown else float(
                            st.number_input(label, value=0, step=1,
                                            key=f"dot_{feat}"))
                    else:
                        dot_values[feat] = tri_state_input(label, key=f"dot_{feat}")

        submitted = st.form_submit_button(
            "Vypočítať kombinované skóre (P3)", type="primary",
            use_container_width=True)

    if submitted:
        X_p3    = make_p3_vector(st.session_state["case_ana_values"], dot_values)
        prob_p3 = predict_p3(X_p3)
        st.session_state["case_dot_values"] = dot_values
        st.session_state["case_prob_p3"]    = prob_p3
        st.session_state["case_step"]       = "3_results"
        st.rerun()


# =============================================================================
# STEP 3 — RESULTS  (P3 hlavný, P1 doplnkový)
# =============================================================================
elif step == "3_results":
    if "case_prob_p3" not in st.session_state:
        st.info("Výsledok sa zobrazí po vyplnení dotazníka.")
        st.stop()

    prob_p1 = st.session_state["case_prob_p1"]
    prob_p3 = st.session_state["case_prob_p3"]

    st.markdown("### 3. Modelový odhad výsledku HUTT testu")

    # --- premenné ---
    label_p3, color_key_p3, headline_p3 = band(prob_p3)
    color_p3  = band_color(color_key_p3)
    thr_p3    = float(pkg_p3.get("threshold", 0.5))
    thr_p1    = float(pkg_p1.get("threshold", 0.5))
    auc_p3    = pkg_p3.get("AUC_CV_mean")
    above_thr = prob_p3 >= thr_p3
    thr_text  = ("nad" if above_thr else "pod")
    label_p1, color_key_p1, _ = band(prob_p1)
    color_p1  = band_color(color_key_p1)
    delta     = prob_p3 - prob_p1
    pred_p1_bin = "pozitívna" if prob_p1 >= thr_p1 else "negatívna"
    pred_p3_bin = "pozitívna" if prob_p3 >= thr_p3 else "negatívna"

    col_main, col_side = st.columns([1.35, 1.0], gap="large")

    with col_main:
        # P3 hlavná karta
        st.markdown(f"""
<div class="clinical-card">
    <div class="metric-title">Hlavný model — P3 (anamnéza + dotazník, 13 premenných)</div>
    <div class="metric-big" style="color:{color_p3};">{prob_p3*100:.1f}%</div>
    <div class="progress-track">
        <div class="progress-fill" style="width:{prob_p3*100:.1f}%; background:{color_p3};"></div>
    </div>
    <span class="badge {badge_class(color_key_p3)}">{headline_p3}</span>
    <div class="metric-sub" style="margin-top:10px;">
        Skóre je <b>{thr_text}</b> interným prahom {thr_p3:.2f} &nbsp;·&nbsp;
        AUC {round(auc_p3*100,1) if auc_p3 else '–'}%
    </div>
    <div style="margin-top:8px; font-size:0.90rem; color:#1a3a5a;">
        {clinical_interpretation(prob_p3)}
    </div>
</div>
""", unsafe_allow_html=True)


    with col_side:
        # P1 karta
        st.markdown(f"""
<div class="soft-card">
    <div class="metric-title">Predbežný model P1 — 5 základných údajov</div>
    <div style="font-size:2.0rem; font-weight:800; color:{color_p1}; margin-top:6px;">{prob_p1*100:.1f}%</div>
    <div class="progress-track">
        <div class="progress-fill" style="width:{prob_p1*100:.1f}%; background:{color_p1};"></div>
    </div>
    <span class="badge {badge_class(color_key_p1)}" style="font-size:0.78rem;">{label_p1}</span>
    <div class="small-muted" style="margin-top:6px;">Posun oproti P3: <b>{delta*100:+.1f} p. b.</b></div>
</div>
""", unsafe_allow_html=True)

        # zhoda / nesúlad
        if pred_p1_bin == pred_p3_bin:
            st.success(
                f"Oba modely predikujú **{pred_p3_bin}** predikciu. "
                "Zhoda zvyšuje konzistentnosť výstupu."
            )
        else:
            st.warning(
                f"Modely sa líšia: P1 → **{pred_p1_bin}**, P3 → **{pred_p3_bin}**. "
                "Rozhodujúci je P3. Interpretujte v klinickom kontexte."
            )



    _exp_l, _exp_r = st.columns([1.35, 1.0], gap="large")
    with _exp_l:
        with st.expander("Čo modelové skóre znamená?"):
            st.markdown(
                "Skóre vyjadruje pravdepodobnosť pozitívneho výsledku HUTT testu podľa modelu. "
                "Na porovnanie so skutočným výsledkom HUTT sa skóre prevádza cez rozhodovací prah "
                "na binárnu predikciu (pozitívna / negatívna).\n\n"
                "| Skóre | Farba | Interpretácia |\n"
                "|---|---|---|\n"
                "| < 40 % | 🟢 zelená | Skôr negatívny výsledok |\n"
                "| 40 – 60 % | 🟡 žltá | Hraničná / neistá zóna |\n"
                "| 60 – 75 % | 🟠 oranžová | Zvýšená pravdepodobnosť pozitívneho výsledku |\n"
                "| ≥ 75 % | 🔴 červená | Výraznejšia pravdepodobnosť pozitívneho výsledku |\n\n"
                "*Skóre odráža riziko v kontexte pacientov odoslaných na HUTT.*"
            )

    with st.expander("Model Card"):
        _n_p1_feats  = len(pkg_p1.get('selected_features', []))
        _n_p3_total  = len(pkg_p3.get('selected_features', []))
        _n_p3_dot    = len(P3_DOT_FEATS)
        st.markdown(f"""
**Účel:** Orientačný odhad modelového skóre pre pozitívny HUTT test (A10=1).

**Cieľová premenná:** A10 (0 = HUTT negatívny, 1 = HUTT pozitívny).

**P1 (anamnestický model):** Extra Trees · {_n_p1_feats} premenných (pohlavie, vek, TK, pulz)
· AUC {round(pkg_p1.get('AUC_CV_mean',0)*100,1)}% ± {round(pkg_p1.get('AUC_std',0)*100,1)}%
· Sens {round(pkg_p1.get('sensitivity_at_thr',0)*100,0):.0f}% / Spec {round(pkg_p1.get('specificity_at_thr',0)*100,0):.0f}%
· **Prah {THRESHOLD_P1:.2f}** (fixný prah pre všetky modely; metriky sú počítané pri 0.5, klinicky orientačné).

**P3 (kombinovaný model):** Extra Trees · {_n_p3_total} premenných
(5 anamnestických + {_n_p3_dot} dotazníkových, výber ConsensusFS z CV)
· AUC {round(pkg_p3.get('AUC_CV_mean',0)*100,1)}% ± {round(pkg_p3.get('AUC_std',0)*100,1)}%
· Sens {round(pkg_p3.get('sensitivity_at_thr',0)*100,0):.0f}% / Spec {round(pkg_p3.get('specificity_at_thr',0)*100,0):.0f}%
· **Prah {THRESHOLD_P3:.2f}** (fixný prah pre všetky modely; metriky sú počítané pri 0.5, klinicky orientačné).

**Feature selection:** ConsensusFS: premenná je zaradená ak ju vybrali aspoň 2 z 3 metód
(Chi² p < 0.05, RF importance nad priemerom, RFE top √n). Výsledná sada je stabilizovaná
konsenzusom aspoň 3 z 5 CV foldov.

**Validácia:** interná 5-fold stratifikovaná CV. Externá validácia nebola vykonaná.

**Trénovacia vzorka:** n = {pkg_p3.get('n_train','?')} pacientov ({pkg_p3.get('n_pos','?')} HUTT+,
prevalencia {round(pkg_p3.get('n_pos',0)/pkg_p3.get('n_train',1)*100) if pkg_p3.get('n_train') else '?'}%).

**Interpretácia skóre:** < 40% = skôr negatívny · 40–60% = hraničná/neistá zóna · 60–75% = zvýšená pravdepodobnosť · ≥ 75% = výraznejšia pravdepodobnosť.
Tieto pásma sú orientačné, klinické rozhodnutie patrí lekárovi.

**Limitácie:** retrospektívne dáta, jednocentrický pôvod, bez externej validácie,
class_weight=balanced (model kalibrovaný pre rovnomernú váhu tried).
Model bol trénovaný prevažne na pacientoch so synkopou a presynkopou. Podskupiny po resuscitácii (n=2) a po epileptickom záchvate (n=8) sú v dátach zastúpené minimálne, preto predikcie pre týchto pacientov treba interpretovať s mimoriadnou opatrnosťou, model pre ne nemá dostatočnú oporu v tréningových dátach.
""")

    st.markdown("""
<div class="warning-box">
Model nebol externe prospektívne validovaný. Výstup predstavuje výskumný odhad
a nie je určený ako samostatný podklad klinického rozhodnutia.
Ide o doplnkovú informáciu k odbornému posúdeniu.
</div>
""", unsafe_allow_html=True)



st.divider()
st.caption("Bakalárska práca · HUTT predikcia · Výskumný prototyp · Streamlit")
