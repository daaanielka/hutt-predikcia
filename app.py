# -*- coding: utf-8 -*-
"""
app_10d_demo.py  –  Demo aplikácia pre lekárov (dvojkrokový prístup)
Predikcia výsledku HUTT testu (krátkodobá strata vedomia)
Bakalárska práca – Daniela

Krok 1: Anamnestické údaje (5 polí) → predbežný výsledok
Krok 2: Dotazníkové otázky (dynamický počet podľa modelu) → spresnený výsledok + porovnanie

Spustenie: streamlit run app_10d_demo.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Nastavenia stránky ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="HUTT Prediktor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Globálny CSS – svetlý aj tmavý režim ─────────────────────────────────────
st.markdown("""
<style>
:root {
    --card-bg:     #fafafa;
    --card-border: #e0e0e0;
    --text-muted:  #666;
    --text-faint:  #999;
    --track-bg:    #e0e0e0;
    --info-bg:     #eef2f7;
    --info-text:   #1a1a2e;
}
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg:     #1e2030;
        --card-border: #3d4058;
        --text-muted:  #b0b8d0;
        --text-faint:  #7880a0;
        --track-bg:    #3d4058;
        --info-bg:     #1a2535;
        --info-text:   #d0d8f0;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Cesty k modelom ─────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
PATH_ANA = os.path.join(BASE, "model_10f_anamneza.joblib")
PATH_KOM = os.path.join(BASE, "model_10f_kombinacia.joblib")

REQUIRED_KEYS = {"pipeline", "features", "threshold", "model_name", "AUC_CV"}

@st.cache_resource
def load_models():
    ana = joblib.load(PATH_ANA)
    kom = joblib.load(PATH_KOM)
    # Schema check – overi že model obsahuje všetky povinné kľúče
    for label, pkg in [("Anamnéza", ana), ("Kombinácia", kom)]:
        missing = REQUIRED_KEYS - set(pkg.keys())
        if missing:
            raise ValueError(f"Model {label} neobsahuje kľúče: {missing}. "
                             f"Pretrénujte model (10f_hutt_predikcia.py).")
    return ana, kom

try:
    pkg_ana, pkg_kom = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Chyba pri načítaní modelov: {e}")
    models_loaded = False
    pkg_ana, pkg_kom = None, None

# ── Vybrané dotazníkové atribúty – načítané priamo z modelu ────────────────
# Zabezpečuje konzistenciu: ak sa model pretrénuje, zoznam sa automaticky aktualizuje
SELECTED_DOT = pkg_kom.get("selected_dot_features", []) if models_loaded else []
N_DOT = len(SELECTED_DOT)
N_ANA = pkg_ana.get("n_ana", 5) if models_loaded else 5  # počet anamnestických premenných

# ── Popisky otázok (text z dotazníka) ──────────────────────────────────────
OTAZKY = {
    # Blok B – Dôvod vyšetrenia
    "B":    "B – Dôvod vyšetrenia (strata vedomia alebo iný stav)?",
    "B1":   "B1 – Strata vedomia (dôvod vyšetrenia)?",
    "B2":   "B2 – Pocity hroziacej straty vedomia?",
    "B3":   "B3 – Stav po resuscitácii?",
    "B4":   "B4 – Stav po epileptickom záchvate?",
    "B5":   "B5 – Opakované pády?",
    # Blok C – Vznik ťažkostí (C1/C3/C4 = vek pri udalosti, C2 = počet)
    "C":    "C – Pacient uviedol konkrétne ťažkosti?",
    "C1":   "C1 – Vek pri prvom výskyte ťažkostí",
    "C2":   "C2 – Celkový počet odpadnutí",
    "C3":   "C3 – Vek pri poslednom odpadnutí",
    "C4":   "C4 – Vek v období najhorších ťažkostí",
    # Blok D – Situácie vedúce k synkope
    "D":    "D – Strata vedomia bola vyvolaná provokujúcim faktorom?",
    "D1":   "D1 – Strata vedomia pri státí?",
    "D2":   "D2 – Strata vedomia do 1 minúty po postavení sa?",
    "D3":   "D3 – Strata vedomia pri chôdzi?",
    "D4":   "D4 – Strata vedomia pri fyzickej námahe?",
    "D5":   "D5 – Strata vedomia v sede?",
    "D6":   "D6 – Strata vedomia poležačky?",
    # Blok E – Faktory vedúce k strate vedomia
    "E":    "E – Strata vedomia bola vyvolaná konkrétnym faktorom?",
    "E1":   "E1 – Strata vedomia v preľudnených priestoroch?",
    "E2":   "E2 – Strata vedomia v dusnom prostredí?",
    "E3":   "E3 – Strata vedomia v teplom prostredí?",
    "E4":   "E4 – Strata vedomia po pohľade na krv?",
    "E5":   "E5 – Strata vedomia po nepríjemných emóciách (strach, úzkosť, rozrušenie)?",
    "E6":   "E6 – Strata vedomia pri medicínskom výkone?",
    "E7":   "E7 – Strata vedomia po bolesti?",
    "E8":   "E8 – Strata vedomia pri dehydratácii?",
    "E9":   "E9 – Strata vedomia počas menštruácie?",
    "E10":  "E10 – Strata vedomia pri strate krvi?",
    # Blok F – Špecifické situácie spojené so stratou vedomia
    "F":    "F – Strata vedomia pri špecifickej situácii?",
    "F1":   "F1 – Strata vedomia pri stolici?",
    "F2":   "F2 – Strata vedomia pri močení?",
    "F3":   "F3 – Strata vedomia pri kašli?",
    "F4":   "F4 – Strata vedomia pri kýchaní / smrkaní?",
    "F5":   "F5 – Strata vedomia pri jedení / prehĺtaní?",
    "F6":   "F6 – Strata vedomia po náhlej bolesti?",
    "F7":   "F7 – Strata vedomia počas fyzickej námahy?",
    "F8":   "F8 – Strata vedomia pri hlade?",
    "F9":   "F9 – Strata vedomia pri nedostatku spánku / únave?",
    "F10":  "F10 – Strata vedomia v inej špecifickej situácii?",
    # Blok G – Lieky alebo alkohol
    "G":    "G – Užitie liekov alebo alkoholu hodinu pred stratou vedomia?",
    # Blok H – Symptómy pred stratou vedomia
    "H":    "H – Príznaky tesne pred stratou vedomia?",
    "H1":   "H1 – Nevoľnosť / pocit na zvracanie pred stratou vedomia?",
    "H2":   "H2 – Pocit tepla / horúčavy pred stratou vedomia?",
    "H3":   "H3 – Potenie (pot) pred stratou vedomia?",
    "H4":   "H4 – Zahmlievanie pred očami pred stratou vedomia?",
    "H5":   "H5 – Hučanie v ušiach pred stratou vedomia?",
    "H6":   "H6 – Búšenie srdca pred stratou vedomia (1)?",
    "H7":   "H7 – Búšenie srdca pred stratou vedomia (2)?",
    "H8":   "H8 – Bolesť na hrudníku pred stratou vedomia?",
    "H9":   "H9 – Neobvyklý zápach pred stratou vedomia?",
    "H10":  "H10 – Neobvyklé zvuky pred stratou vedomia?",
    "H11":  "H11 – Poruchy reči alebo slabosť polovice tela pred stratou vedomia?",
    "H12":  "H12 – Žiadne zvláštne pocity pred stratou vedomia?",
    "H13":  "H13 – Nepamätám sa na pocity pred stratou vedomia?",
    "H14":  "H14 – Iné príznaky pred stratou vedomia?",
    # Blok I – Trvanie symptómov pred stratou vedomia
    "I":    "I – Príznaky pred stratou vedomia trvali istú dobu?",
    "I1":   "I1 – Príznaky trvali niekoľko sekúnd?",
    "I2":   "I2 – Príznaky trvali do 1 minúty?",
    "I3":   "I3 – Príznaky trvali do 5 minút?",
    "I4":   "I4 – Príznaky trvali viac ako 5 minút?",
    # Blok J – Reakcia pacienta pred stratou vedomia
    "J":    "J – Pacient reagoval pri hroziacej strate vedomia?",
    "J2":   "J2 – Ľahol si pri hroziacej strate vedomia?",
    "J3":   "J3 – Nestihol nič urobiť pred stratou vedomia?",
    # Blok K – Trvanie bezvedomia podľa svedkov
    "K":    "K – Trvanie bezvedomia podľa svedkov?",
    "K1":   "K1 – Bezvedomie trvalo niekoľko sekúnd (podľa svedkov)?",
    "K2":   "K2 – Bezvedomie trvalo do 1 minúty (podľa svedkov)?",
    "K3":   "K3 – Bezvedomie trvalo do 5 minút (podľa svedkov)?",
    "K4":   "K4 – Bezvedomie trvalo viac ako 5 minút (podľa svedkov)?",
    # Blok L, M
    "L":    "L – Kŕče počas bezvedomia?",
    "M":    "M – Inkontinencia (stolica alebo moč) počas bezvedomia?",
    # Blok N – Stav po prebudení
    "N":    "N – Pamäť na udalosti po strate vedomia?",
    "N1":   "N1 – Pohryzený jazyk alebo pery po strate vedomia?",
    "N2":   "N2 – Poranenie / úder pri páde?",
    "N3":   "N3 – Dezorientovanosť viac ako 30 minút po prebratí (podľa svedkov)?",
    "N4":   "N4 – Bolesti hlavy alebo svalov po prebratí?",
    "N5":   "N5 – Nevoľnosť po prebratí?",
    "N6":   "N6 – Cítil/a sa normálne po prebratí (bez ťažkostí)?",
    "N7":   "N7 – Nepamätá sa na stav po prebratí?",
    # Blok O – Rodinná anamnéza
    "O":    "O – Výskyt ochorení v rodine?",
    "O1":   "O1 – Náhle úmrtie člena rodiny?",
    "O2":   "O2 – Ochorenie srdca v rodine (1)?",
    "O3":   "O3 – Ochorenie srdca v rodine (2)?",
    "O4":   "O4 – Srdcová arytmia / kardiostimulátor v rodine?",
    "O5":   "O5 – Ochorenie mozgu / epilepsia v rodine?",
    # Blok P – Osobná anamnéza
    "P":    "P – Osobná anamnéza – liečené ochorenia?",
    "P1":   "P1 – Ochorenie srdca (1) v osobnej anamnéze?",
    "P2":   "P2 – Ochorenie srdca (2) v osobnej anamnéze?",
    "P3":   "P3 – Ochorenie chlopní v osobnej anamnéze?",
    "P4":   "P4 – Srdcová slabosť (srdcové zlyhávanie) v osobnej anamnéze?",
    "P5":   "P5 – Koronárna choroba srdca v osobnej anamnéze?",
    "P6":   "P6 – Srdcové arytmie v osobnej anamnéze?",
    "P7":   "P7 – Búšenie srdca v osobnej anamnéze?",
    "P9":   "P9 – Bolesti na hrudníku v osobnej anamnéze?",
    "P10":  "P10 – Vysoký tlak krvi (hypertenzia) v osobnej anamnéze?",
    "P11":  "P11 – Nízky tlak krvi (hypotenzia) v osobnej anamnéze?",
    "P12":  "P12 – Závraty v osobnej anamnéze?",
    "P13":  "P13 – Ochorenia obličiek v osobnej anamnéze?",
    "P14":  "P14 – Diabetes (cukrovka) v osobnej anamnéze?",
    "P15":  "P15 – Anémia (chudokrvnosť) v osobnej anamnéze?",
    "P16":  "P16 – Astma v osobnej anamnéze?",
    "P17":  "P17 – Ochorenia pľúc v osobnej anamnéze?",
    "P18":  "P18 – Ochorenia priedušiek v osobnej anamnéze?",
    "P19":  "P19 – Ochorenia žalúdka v osobnej anamnéze?",
    "P20":  "P20 – Ochorenia čreva v osobnej anamnéze?",
    "P21":  "P21 – Ochorenia štítnej žľazy v osobnej anamnéze?",
    "P22":  "P22 – Endokrinologické ochorenia v osobnej anamnéze?",
    "P23":  "P23 – Bolesti hlavy v osobnej anamnéze?",
    "P24":  "P24 – Neurologické ochorenia v osobnej anamnéze?",
    "P25":  "P25 – Parkinsonova choroba v osobnej anamnéze?",
    "P26":  "P26 – Psychiatrické ochorenia v osobnej anamnéze?",
    "P27":  "P27 – Depresia v osobnej anamnéze?",
    "P28":  "P28 – Ochorenia krčnej chrbtice v osobnej anamnéze?",
    "P29":  "P29 – Bolesti chrbta v osobnej anamnéze?",
    "P30":  "P30 – Reumatologické ochorenia v osobnej anamnéze?",
    "P31":  "P31 – Nádorové ochorenie v osobnej anamnéze?",
    "P32":  "P32 – Prekonané operácie v osobnej anamnéze?",
    "P33":  "P33 – Prekonané úrazy v osobnej anamnéze?",
    "P34":  "P34 – Alergie v osobnej anamnéze?",
    # Blok Q – Predchádzajúce vyšetrenia
    "Q":    "Q – Predchádzajúce vyšetrenia kvôli stratám vedomia?",
    "Q2":   "Q2 – Záťažový test (bicyklová ergometria)?",
    "Q3":   "Q3 – Koronografické vyšetrenie?",
    "Q5":   "Q5 – Pažerákova stimulácia?",
    "Q6":   "Q6 – Invazívne vyšetrenie arytmií (EFV)?",
    "Q7":   "Q7 – Nukleárne vyšetrenie srdca (SPECT)?",
    "Q8":   "Q8 – CT srdca?",
    "Q9":   "Q9 – MRI srdca?",
    "Q10":  "Q10 – Neurologické vyšetrenia?",
    "Q11":  "Q11 – USG mozgových ciev?",
    "Q14":  "Q14 – Elektromyografia (EMG)?",
    "Q15":  "Q15 – Psychiatrické vyšetrenie?",
    # Blok R – Očkovania
    "R1":   "R1 – Očkovanie proti HPV?",
    "R2":   "R2 – Očkovanie proti chrípke?",
    # Odvodená premenná
    "Ma_diag_srdcove_ochorenie": "Diagnostikované srdcové ochorenie (P1–P8)?",
}

# ── Pomocné funkcie ──────────────────────────────────────────────────────────
def predict(pkg, X_input):
    prob = pkg["pipeline"].predict_proba(X_input)[0, 1]
    pred = int(prob >= pkg["threshold"])
    return prob, pred

_BAND_MARGIN = 0.10  # pásmo neistoty okolo prahu = prah ± MARGIN

def score_band(prob, threshold=0.45):
    """
    Tri pásma dynamicky podľa prahu modelu.
      zvýšené  : prob >= threshold + MARGIN
      hraničné : threshold - MARGIN <= prob < threshold + MARGIN
      nízke    : prob < threshold - MARGIN
    Takto sa hraničné pásmo vždy stretáva s rozhodovacím prahom.
    """
    if prob >= threshold + _BAND_MARGIN:   return "zvýšené",  "#e74c3c"
    elif prob >= threshold - _BAND_MARGIN: return "hraničné", "#e67e22"
    else:                                  return "nízke",    "#27ae60"


def gauge_html(prob, label, subtext="", threshold=0.45):
    pct  = int(prob * 100)
    band, col = score_band(prob, threshold)
    return f"""
    <div style='text-align:center; padding:12px; background:var(--card-bg);
                border-radius:10px; border:1px solid var(--card-border);'>
      <div style='font-size:0.85em; color:var(--text-muted); margin-bottom:4px;'>{label}</div>
      <div style='font-size:3em; font-weight:bold; color:{col}; line-height:1.1;'>{pct}%</div>
      <div style='background:var(--track-bg); border-radius:8px; height:12px; margin:8px 4px;'>
        <div style='background:{col}; width:{pct}%; height:12px;
                    border-radius:8px; transition:width 0.6s;'></div>
      </div>
      <div style='font-size:0.82em; font-weight:bold; color:{col}; margin-top:4px;'>
        Pásmo: {band}</div>
      <div style='font-size:0.78em; color:var(--text-faint);'>{subtext}</div>
    </div>"""

def verdict_html(prob, threshold=0.45):
    band, col = score_band(prob, threshold)
    if band == "zvýšené":
        text = "🔴 Zvýšené modelové skóre — odporúča sa klinické posúdenie"
    elif band == "hraničné":
        text = "🟠 Hraničné modelové skóre — výsledok je neistý, zvážte doplňujúce vyšetrenie"
    else:
        text = "🟢 Nízke modelové skóre — nižší orientačný odhad rizika pozitívneho HUTT testu"
    return (f"<div style='text-align:center; color:{col}; font-size:1.0em; "
            f"font-weight:bold; margin-top:6px;'>{text}</div>")

def tristate(label, key):
    """
    Tri stavy: Neznáme = NaN (imputácia), Áno = 1.0, Nie = 0.0.
    Ak symptóm nebol prítomný → zvoľte Nie. Ak neviete → Neznáme.
    """
    opt = st.radio(label, ["❓ Neznáme", "✅ Áno", "☐ Nie"],
                   horizontal=True, key=key, index=0)
    if opt == "✅ Áno": return 1.0
    if opt == "☐ Nie":  return 0.0
    return np.nan

def build_kom_input(pohlavie_enc, vek, tk_sys, tk_dia, pulz, dotaznik_vals):
    """Vytvorí vstupný vektor pre Kombinacia model (všetky features ako NaN, vyplní známe)."""
    feats = pkg_kom["features"]
    row = {f: np.nan for f in feats}
    row["Pohlavie_enc"] = pohlavie_enc
    row["Vek"]          = vek
    row["TK_sys"]       = tk_sys
    row["TK_dia"]       = tk_dia
    row["Pulz"]         = pulz
    for kod, val in dotaznik_vals.items():
        if kod in row:
            row[kod] = val
    return np.array([[row[f] for f in feats]])

def vyplnenost(dotaznik_vals, numeric_keys):
    """Počet vyplnených otázok (nie Neznáme / nie NaN pre C1/C2/C4)."""
    vyplnene = 0
    for kod, val in dotaznik_vals.items():
        if kod in numeric_keys:
            # NaN = neznáme (analýza konvertuje -1→NaN pred tréningom)
            if val is not None and not np.isnan(float(val)):
                vyplnene += 1
        else:
            if val in (0.0, 1.0):
                vyplnene += 1
    return vyplnene

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=65)
    st.title("HUTT Prediktor")
    st.markdown(f"""
    **Predikcia výsledku HUTT testu**
    Bakalárska práca · 2025/2026

    ---
    **Dvojkrokový prístup:**

    **Krok 1** – Anamnéza *(5 polí)*
    Rýchly odhad z klinických meraní

    **Krok 2** – Dotazník *({N_DOT} otázok)*
    Spresnený výsledok s anamnézou

    ---
    **Modely:**
    - 🟦 **Anamnéza / {pkg_ana.get('model_name','ExtraTrees')}**
      AUC_CV = {pkg_ana.get('AUC_CV','?')}% ± {pkg_ana.get('AUC_CV_std','?')}% · 5 premenných
    - 🟩 **Kombinácia / {pkg_kom.get('model_name','RF')}**
      AUC_CV = {pkg_kom.get('AUC_CV','?')}% ± {pkg_kom.get('AUC_CV_std','?')}% · 5+{N_DOT} premenných
      *(skriningový prah={pkg_kom.get('threshold',0.30):.2f}: senzit. 95%, špecif. ~45%)*

    ---
    **Tri pásma modelového skóre** *(relatívne k prahovej hodnote modelu)*:
    🟢 nízke — skóre výrazne pod prahom
    🟠 hraničné — skóre v okolí prahu (±10 %)
    🔴 zvýšené — skóre výrazne nad prahom

    **Cieľ:** orientačný odhad výsledku HUTT testu

    ---
    ⚠️ *Orientačný prototyp.
    Nenahradzuje klinické rozhodnutie.*
    """)
    st.markdown("---")
    if st.button("🔄 Nový pacient", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            if key.startswith("q2_") or key in {
                "prob_ana", "pred_ana", "ana_inputs", "step2_open",
                "step2_done", "prob_kom", "pred_kom", "dotaznik_vals",
                "n_vyplnene", "_last_page",
            }:
                st.session_state.pop(key, None)
        st.session_state["page"] = 0
        st.rerun()

# ── Hlavička ─────────────────────────────────────────────────────────────────
st.title("🫀 Predikcia výsledku HUTT testu")
st.markdown(f"""
Zadajte údaje pacienta v **dvoch krokoch**.
Krok 1 je povinný, Krok 2 je voliteľný a spresňuje **orientačné modelové skóre** pomocou {N_DOT} otázok z dotazníka.
Výstup je výskumný prototyp — nenahradzuje klinické rozhodnutie.
""")

if not models_loaded:
    st.error("Modely sa nenačítali. Skontrolujte súbory model_10f_*.joblib")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# NAVIGÁCIA (namiesto st.tabs – programatické prepínanie cez session_state)
# ════════════════════════════════════════════════════════════════════════════
_ana_name = pkg_ana.get('model_name', 'ExtraTrees')
_kom_name = pkg_kom.get('model_name', 'RF')

if "page" not in st.session_state:
    st.session_state["page"] = 0

_nav1, _nav2, _nav3 = st.columns(3)
with _nav1:
    if st.button("1 — Anamnéza", use_container_width=True,
                 type="primary" if st.session_state["page"] == 0 else "secondary"):
        st.session_state["page"] = 0
        st.rerun()
with _nav2:
    if st.button(f"2 — Dotazník ({N_DOT} otázok)", use_container_width=True,
                 type="primary" if st.session_state["page"] == 1 else "secondary"):
        st.session_state["page"] = 1
        st.rerun()
with _nav3:
    if st.button("3 — Výsledky", use_container_width=True,
                 type="primary" if st.session_state["page"] == 2 else "secondary"):
        st.session_state["page"] = 2
        st.rerun()
st.markdown("---")

page = st.session_state["page"]

# Scroll nahor pri každej zmene strany
if st.session_state.get("_last_page") != page:
    st.session_state["_last_page"] = page
    components.html("""
    <script>
    (function() {
        var el = window.parent.document.querySelector('section[data-testid="stMain"]')
              || window.parent.document.querySelector('.main')
              || window.parent.document.body;
        if (el) el.scrollTop = 0;
        window.parent.scrollTo(0, 0);
    })();
    </script>
    """, height=0)

# ════════════════════════════════════════════════════════════════════════════
# STRANA 1 – ANAMNESTICKÉ ÚDAJE
# ════════════════════════════════════════════════════════════════════════════
if page == 0:
    st.markdown("## Krok 1 — Anamnestické údaje")
    st.caption("Základné klinické merania dostupné pred HUTT testom")

    col1, col2, col3 = st.columns(3)
    with col1:
        pohlavie     = st.selectbox("Pohlavie", ["Žena", "Muž"])
        pohlavie_enc = 1.0 if pohlavie == "Muž" else 0.0
        vek          = st.number_input("Vek (roky)", min_value=1, max_value=110, value=45, step=1)
    with col2:
        tk_sys = st.number_input("TK systolický (mmHg)", min_value=60, max_value=250, value=120, step=1)
        tk_dia = st.number_input("TK diastolický (mmHg)", min_value=30, max_value=150, value=80, step=1)
    with col3:
        pulz = st.number_input("Pulz (tepy/min)", min_value=20, max_value=200, value=70, step=1)

    btn_krok1 = st.button("🔍 Vypočítaj predbežný výsledok", type="primary",
                           use_container_width=True)

    if btn_krok1:
        X_ana = np.array([[pohlavie_enc, vek, tk_sys, tk_dia, pulz]])
        prob_ana, pred_ana = predict(pkg_ana, X_ana)
        st.session_state["prob_ana"]   = prob_ana
        st.session_state["pred_ana"]   = pred_ana
        st.session_state["ana_inputs"] = (pohlavie_enc, vek, tk_sys, tk_dia, pulz)
        st.session_state["step2_done"] = False

    if "prob_ana" in st.session_state:
        prob_ana = st.session_state["prob_ana"]
        st.markdown("---")
        st.markdown(f"### Predbežný výsledok — Anamnéza / {_ana_name}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(gauge_html(prob_ana, f"Anamnéza / {_ana_name}",
                                   f"prah={pkg_ana['threshold']:.2f} · 5 premenných",
                                   threshold=pkg_ana['threshold']),
                        unsafe_allow_html=True)
            st.markdown(verdict_html(prob_ana, threshold=pkg_ana['threshold']),
                        unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style='padding:12px; background:var(--info-bg); color:var(--info-text); border-radius:8px; margin-top:10px;'>
            <b>Čo tento výsledok znamená?</b><br><br>
            Model vypočítal <b>orientačné modelové skóre</b> na základe
            <b>5 základných klinických meraní</b> (vek, pohlavie, TK, pulz).<br><br>
            Skóre nie je klinicky validovaná pravdepodobnosť — ide o výskumný odhad.
            Pre spresnenie prejdite na záložku <b>2 — Dotazník</b>.
            </div>
            """, unsafe_allow_html=True)
        st.caption(f"AUC_CV = {pkg_ana['AUC_CV']}% ± {pkg_ana.get('AUC_CV_std','?')}%  ·  "
                   f"Senzitivita=93% · Špecificita=23%")
        st.markdown("---")
        if st.button("📝 Prejsť na dotazník →", use_container_width=True):
            st.session_state["page"] = 1
            st.rerun()
    else:
        st.info("Vyplňte údaje vyššie a stlačte **Vypočítaj predbežný výsledok**.")

# ════════════════════════════════════════════════════════════════════════════
# STRANA 2 – DOTAZNÍK
# ════════════════════════════════════════════════════════════════════════════
if page == 1:
    if "prob_ana" not in st.session_state:
        st.warning("⚠️ Najprv vyplňte **Krok 1 — Anamnéza** a vypočítajte predbežný výsledok.")
    else:
        st.markdown(f"## Krok 2 — Dotazníkové otázky ({N_DOT} otázok)")
        st.caption(
            "Odpovede: ✅ = Áno · ☐ = Nie · ❓ = Neznáme  |  "
            "Neznáme hodnoty sú nahradené mediánom z trénovacej vzorky. "
            "Pri väčšom počte neznámych odpovedí je výsledok menej spoľahlivý."
        )
        st.caption("💡 **Návod:** Ak symptóm nebol prítomný → zvoľte **Nie**. "
                   "Ak informácia nie je dostupná → zvoľte **Neznáme**. "
                   "Číselné polia C1, C2, C4 nechajte prázdne ak hodnota nie je známa.")

        NUMERIC_INPUTS = {
            "C1": {"label": "C1 – Vek pri prvom výskyte ťažkostí (roky)",
                   "min": 1, "max": 100, "default": 1, "step": 1},
            "C2": {"label": "C2 – Celkový počet odpadnutí (prázdne = neznáme)",
                   "min": 0, "max": 200, "default": None, "step": 1},
            "C4": {"label": "C4 – Vek v období najhorších ťažkostí (roky)",
                   "min": 1, "max": 100, "default": 1, "step": 1},
        }
        NUMERIC_KEYS = set(NUMERIC_INPUTS.keys())

        BLOKY = {
            "C – Vznik ťažkostí":                    ["C1", "C2", "C4"],
            "D – Situácie vedúce k synkope":          ["D2"],
            "E – Faktory vedúce k strate vedomia":    ["E", "E1", "E4", "E5", "E7"],
            "F – Špecifické situácie":                ["F1", "F2", "F4"],
            "H – Symptómy pred stratou vedomia":      ["H", "H1", "H3", "H5", "H13"],
            "I – Trvanie symptómov pred synkopou":    ["I1", "I2"],
            "K – Trvanie bezvedomia (podľa svedkov)": ["K", "K4"],
            "N – Stav po prebudení":                  ["N1", "N6"],
            "O – Rodinná anamnéza":                   ["O1"],
            "P – Osobná anamnéza":                    ["P5", "P9", "P18", "P27", "P31", "P33"],
        }

        # ── Formulár – žiadny rerender pri vypĺňaní, iba pri odoslaní ────────
        dotaznik_vals = {}
        with st.form("dotaznik_form"):
            for blok_nazov, kody in BLOKY.items():
                kody_v_modeli = [k for k in kody if k in SELECTED_DOT]
                if not kody_v_modeli:
                    continue
                st.markdown(f"**{blok_nazov}**")
                cols = st.columns(2)
                for i, kod in enumerate(kody_v_modeli):
                    with cols[i % 2]:
                        if kod in NUMERIC_INPUTS:
                            cfg = NUMERIC_INPUTS[kod]
                            val = st.number_input(cfg["label"], min_value=cfg["min"],
                                                  max_value=cfg["max"], value=cfg["default"],
                                                  step=cfg["step"], key=f"q2_{kod}")
                            dotaznik_vals[kod] = float(val) if val is not None else np.nan
                        else:
                            label = OTAZKY.get(kod, f"{kod} – [doplňte text otázky]")
                            dotaznik_vals[kod] = tristate(label, f"q2_{kod}")
                st.markdown("")
            btn_krok2 = st.form_submit_button("🎯 Vypočítaj spresnený výsledok",
                                               type="primary", use_container_width=True)

        if btn_krok2:
            n_vyplnene = vyplnenost(dotaznik_vals, NUMERIC_KEYS)
            pohlavie_enc, vek, tk_sys, tk_dia, pulz = st.session_state["ana_inputs"]
            X_kom = build_kom_input(pohlavie_enc, vek, tk_sys, tk_dia, pulz, dotaznik_vals)
            prob_kom, pred_kom = predict(pkg_kom, X_kom)
            st.session_state["prob_kom"]      = prob_kom
            st.session_state["pred_kom"]      = pred_kom
            st.session_state["dotaznik_vals"] = dotaznik_vals
            st.session_state["n_vyplnene"]    = n_vyplnene
            st.session_state["step2_done"]    = True
            st.session_state["page"]          = 2
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STRANA 3 – FINÁLNY VÝSLEDOK
# ════════════════════════════════════════════════════════════════════════════
if page == 2:
    if not st.session_state.get("step2_done"):
        st.info("ℹ️ Výsledky sa zobrazia po vyplnení dotazníka v záložke **2 — Dotazník**.")
        if "prob_ana" in st.session_state:
            st.markdown("---")
            st.markdown("**Zatiaľ dostupný len predbežný výsledok (Anamnéza):**")
            prob_ana = st.session_state["prob_ana"]
            st.markdown(gauge_html(prob_ana, f"Anamnéza / {_ana_name}",
                                   f"prah={pkg_ana['threshold']:.2f}",
                                   threshold=pkg_ana['threshold']),
                        unsafe_allow_html=True)
            st.markdown(verdict_html(prob_ana, threshold=pkg_ana['threshold']),
                        unsafe_allow_html=True)
    else:
        prob_ana   = st.session_state["prob_ana"]
        prob_kom   = st.session_state["prob_kom"]
        n_vyplnene = st.session_state.get("n_vyplnene", N_DOT)
        dot_vals   = st.session_state.get("dotaznik_vals", {})

        _th, _btn_col = st.columns([3, 1])
        with _th:
            st.markdown("## Finálny výsledok — Porovnanie modelov")
        with _btn_col:
            st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
            if st.button("🔄 Nový pacient", use_container_width=True, type="primary"):
                for key in list(st.session_state.keys()):
                    if key.startswith("q2_") or key in {
                        "prob_ana", "pred_ana", "ana_inputs", "step2_open",
                        "step2_done", "prob_kom", "pred_kom", "dotaznik_vals",
                        "n_vyplnene", "_last_page",
                    }:
                        st.session_state.pop(key, None)
                st.session_state["page"] = 0
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        c1, mid, c2 = st.columns([5, 1, 5])
        with c1:
            st.markdown(gauge_html(prob_ana, f"🟦 Anamnéza / {_ana_name}",
                                   f"prah={pkg_ana['threshold']:.2f} · 5 premenných",
                                   threshold=pkg_ana['threshold']),
                        unsafe_allow_html=True)
            st.markdown(verdict_html(prob_ana, threshold=pkg_ana['threshold']),
                        unsafe_allow_html=True)
            st.caption(f"AUC_CV = {pkg_ana['AUC_CV']}% ± {pkg_ana.get('AUC_CV_std','?')}%  ·  "
                       f"Senzitivita=93% · Špecificita=23%")
        with mid:
            st.markdown("<br><br><div style='text-align:center;font-size:1.8em;'>⟷</div>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(gauge_html(prob_kom, f"🟩 Kombinácia / {_kom_name}",
                                   f"prah={pkg_kom['threshold']:.2f} · 5+{N_DOT} premenných",
                                   threshold=pkg_kom['threshold']),
                        unsafe_allow_html=True)
            st.markdown(verdict_html(prob_kom, threshold=pkg_kom['threshold']),
                        unsafe_allow_html=True)
            st.caption(f"AUC_CV = {pkg_kom['AUC_CV']}% ± {pkg_kom.get('AUC_CV_std','?')}%  ·  "
                       f"Senzitivita=95% · Špecificita=45%  ·  "
                       f"dotazník: {n_vyplnene}/{N_DOT} otázok vyplnených")

        # ── Zhoda modelov ────────────────────────────────────────────────────
        st.markdown("---")
        band_ana = score_band(prob_ana, threshold=pkg_ana['threshold'])[0]
        band_kom = score_band(prob_kom, threshold=pkg_kom['threshold'])[0]
        if band_ana == band_kom:
            st.success(f"✅ Oba modely sa **zhodujú** — pásmo: **{band_kom}**")
        else:
            st.warning(
                f"⚠️ Modely sa **nezhodujú** — Anamnéza: **{band_ana}** · Kombinácia: **{band_kom}**\n\n"
                "Kombinovaný model zohľadňuje symptómy z dotazníka. "
                "Pri rozdielnych výsledkoch zvážte oba pohľady; rozhodnutie ostáva na lekárovi."
            )

        # ── Zmena skóre ──────────────────────────────────────────────────────
        delta = prob_kom - prob_ana
        delta_pct = int(abs(delta) * 100)
        if abs(delta) >= 0.05:
            smer = "zvýšil" if delta > 0 else "znížil"
            st.info(f"📊 Dotazník {smer} modelové skóre o **{delta_pct} pp** "
                    f"({int(prob_ana*100)}% → {int(prob_kom*100)}%)")
        else:
            st.info(f"📊 Dotazník zmenil skóre len minimálne ({delta_pct} pp) — modely sú konzistentné.")

        # ── Grafy ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Doplňujúce grafy")
        _v1, _v2 = st.columns(2)

        with _v1:
            st.markdown("**Distribúcia modelového skóre (trénovacie dáta)**")
            st.caption("Kde sa váš pacient nachádza oproti HUTT+ a HUTT− pacientom")
            _pos = pkg_kom.get("train_proba_pos", [])
            _neg = pkg_kom.get("train_proba_neg", [])
            if _pos and _neg:
                _fig, _ax = plt.subplots(figsize=(5, 3.2))
                _bins = np.linspace(0, 1, 21)
                _ax.hist(_neg, bins=_bins, alpha=0.6, color="#27ae60", label=f"HUTT− (n={len(_neg)})",
                         density=True, edgecolor="white", linewidth=0.5)
                _ax.hist(_pos, bins=_bins, alpha=0.6, color="#e74c3c", label=f"HUTT+ (n={len(_pos)})",
                         density=True, edgecolor="white", linewidth=0.5)
                _ax.axvline(prob_kom, color="#f39c12", linewidth=2.5, linestyle="--",
                            label=f"Váš pacient ({int(prob_kom*100)}%)")
                _ax.axvline(pkg_kom["threshold"], color="#e67e22", linewidth=1.5, linestyle=":",
                            label=f"Prah ({int(pkg_kom['threshold']*100)}%)")
                _ax.set_xlabel("Modelové skóre")
                _ax.set_ylabel("Hustota")
                _ax.legend(fontsize=8, loc="upper center")
                _ax.set_xlim(0, 1)
                _ax.spines[['top','right']].set_visible(False)
                _fig.tight_layout()
                st.pyplot(_fig, use_container_width=True)
                plt.close(_fig)
            else:
                st.info("Distribučné dáta nie sú dostupné.")

        with _v2:
            st.markdown("**Senzitivita / Špecificita pri rôznych prahoch**")
            st.caption("Ako sa mení záchytnosť modelu pri zmene rozhodovacieho prahu")
            _prah_data = pkg_kom.get("prah_table", [])
            if _prah_data:
                _prah_df = pd.DataFrame(_prah_data)
                _show_thrs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                _prah_df["thr_r"] = _prah_df["Prah"].round(2)
                _prah_filt = _prah_df[_prah_df["thr_r"].isin(_show_thrs)].copy()
                _prah_filt = _prah_filt[["thr_r", "Sens_%", "Spec_%", "FN", "FP"]].rename(columns={
                    "thr_r": "Prah", "Sens_%": "Sens %", "Spec_%": "Spec %"})
                _thr_r = round(pkg_kom["threshold"], 2)
                def _highlight_row(row):
                    if round(row["Prah"], 2) == _thr_r:
                        return ["background-color: #e67e22; color: white; font-weight: bold"] * len(row)
                    return [""] * len(row)
                st.dataframe(_prah_filt.style.apply(_highlight_row, axis=1),
                             use_container_width=True, hide_index=True)
                st.caption(f"🟡 Žltý riadok = odporúčaný prah ({_thr_r}) · FN = zmeškaní HUTT+")
            else:
                st.info("Tabuľka prahov nie je dostupná.")

        # ── Model Card ───────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📄 Model Card – informácie o modeli"):
            _n_feat   = N_ANA + N_DOT
            _thr_note = pkg_kom.get('threshold_note', '')
            st.markdown(f"""
**Cieľ modelu:** Orientačný odhad pravdepodobnosti pozitívneho výsledku HUTT testu.

**Trénovacie dáta:** n=371 pacientov, jedno centrum (SR), retrospektívna štúdia.
Trénovacia sada: n=297 (80 %) · Testovacia sada: n=74 (20 %).

**Modely:**
- Anamnéza: {pkg_ana.get('model_name','ExtraTrees')} · AUC_CV={pkg_ana.get('AUC_CV','?')}% ± {pkg_ana.get('AUC_CV_std','?')}% · {N_ANA} premenných · prah={pkg_ana.get('threshold','?'):.2f}
- Kombinácia: {pkg_kom.get('model_name','RF')} · AUC_CV={pkg_kom.get('AUC_CV','?')}% ± {pkg_kom.get('AUC_CV_std','?')}% · {_n_feat} premenných · prah={pkg_kom.get('threshold','?'):.2f}

**Výber prahu:** {_thr_note}

**Limitácie:** Interná validácia (1 centrum) · Nested CV na 80 % dát · Malý dataset (n=371) · Kalibrácia neoverená prospektívne.

**Zakázané použitie:** Nenahradzuje klinické vyšetrenie ani rozhodnutie lekára.
            """)

        # ── Disclaimer ───────────────────────────────────────────────────────
        st.info(
            "ℹ️ Výskumný prototyp — nie klinicky validovaný diagnostický nástroj. "
            "Modely validované interne na n=371 pacientoch (n_test=74). "
            "Externá validácia chýba. Výsledok **nenahradzuje klinické rozhodnutie lekára**."
        )

        # ── Súhrn vstupov ────────────────────────────────────────────────────
        with st.expander("📋 Zobraz všetky zadané údaje pacienta"):
            pohlavie_enc, vek, tk_sys, tk_dia, pulz = st.session_state["ana_inputs"]
            df_vstup = pd.DataFrame({
                "Premenná": ["Pohlavie", "Vek", "TK systolický", "TK diastolický", "Pulz"],
                "Hodnota":  ["Muž" if pohlavie_enc == 1 else "Žena",
                             f"{vek} rokov", f"{tk_sys} mmHg", f"{tk_dia} mmHg",
                             f"{pulz} tep/min"]
            })
            st.table(df_vstup.set_index("Premenná"))
            if dot_vals:
                st.markdown(f"**Dotazník ({n_vyplnene}/{N_DOT} vyplnených):**")
                dot_display = []
                for kod, val in dot_vals.items():
                    label = OTAZKY.get(kod, kod)
                    if kod in {"C1", "C2", "C4"}:
                        ans = str(int(val)) if (val is not None and not np.isnan(float(val))) else "Neznáme"
                    else:
                        ans = "Áno" if val == 1.0 else ("Nie" if val == 0.0 else "Neznáme")
                    dot_display.append({"Otázka": label, "Odpoveď": ans})
                st.table(pd.DataFrame(dot_display).set_index("Otázka"))

        # ── Nový pacient ─────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔄 Nový pacient", use_container_width=True, type="primary"):
            for key in ["prob_ana", "pred_ana", "ana_inputs", "step2_done",
                        "prob_kom", "pred_kom", "dotaznik_vals", "n_vyplnene"]:
                st.session_state.pop(key, None)
            st.session_state["page"] = 0  # preskok späť na stranu 1
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Bakalárska práca · 2025/2026 · Python 3 · Streamlit")
