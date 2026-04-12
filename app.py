"""
Predikcia výsledku HUTT testu (synkopa)
Streamlit webová aplikácia – redesign pre klinické použitie
Model: Random Forest – Anamnéza (chi² výber)
"""
import streamlit as st
import numpy as np
import os
import csv
import uuid
from datetime import datetime

# gspread – voliteľná závislosť (len na Streamlit Cloud)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HUTT Predikcia – Synkopa",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Základné štýly */
  .stApp { font-family: 'Segoe UI', Arial, sans-serif; background: #F0F4F8; }

  /* Hlavička */
  .app-header {
    background: linear-gradient(135deg, #0D47A1 0%, #1565C0 60%, #1976D2 100%);
    border-radius: 14px; padding: 22px 28px 18px 28px;
    margin-bottom: 20px; color: white;
    display: flex; align-items: center; gap: 18px;
  }
  .app-header h1 {
    margin: 0; font-size: 1.55rem; font-weight: 700; color: white !important; line-height: 1.2;
  }
  .app-header .subtitle {
    font-size: 0.88rem; color: rgba(255,255,255,0.82); margin-top: 4px;
  }
  .app-header .badge {
    background: rgba(255,255,255,0.18); border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; display: inline-block; margin-top: 6px;
  }

  /* Karty sekcií */
  .card {
    background: white; border-radius: 14px; padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 16px;
  }
  .card-title {
    font-size: 0.95rem; font-weight: 700; color: #0D47A1;
    text-transform: uppercase; letter-spacing: 0.05em;
    border-bottom: 2px solid #E8EEF7; padding-bottom: 10px; margin-bottom: 14px;
  }

  /* Výsledkový box */
  .result-high   { background: linear-gradient(135deg,#FDEDEC,#FBD7D4); border: 2px solid #E74C3C; color: #922B21; }
  .result-medium { background: linear-gradient(135deg,#FEF9E7,#FDE8A0); border: 2px solid #E67E22; color: #7D4608; }
  .result-low    { background: linear-gradient(135deg,#EAFAF1,#C8F5DC); border: 2px solid #27AE60; color: #1D6A39; }
  .result-box {
    border-radius: 14px; padding: 20px 22px; text-align: center;
    margin: 10px 0 16px 0;
  }
  .result-prob { font-size: 3rem; font-weight: 800; line-height: 1; }
  .result-label { font-size: 1.15rem; font-weight: 700; margin: 6px 0 4px 0; }
  .result-desc  { font-size: 0.88rem; opacity: 0.85; }

  /* Klinické odporúčanie */
  .rec-box {
    border-radius: 10px; padding: 14px 18px; margin: 8px 0 16px 0;
    font-size: 0.92rem; font-weight: 500;
  }
  .rec-high   { background: #FFF3F2; border-left: 5px solid #E74C3C; color: #6B1A14; }
  .rec-medium { background: #FFFBF0; border-left: 5px solid #E67E22; color: #5D3208; }
  .rec-low    { background: #F2FBF5; border-left: 5px solid #27AE60; color: #154D2A; }

  /* Gauge lišta */
  .gauge-wrap { margin: 8px 0 4px 0; }
  .gauge-track {
    background: linear-gradient(to right, #27AE60 0%, #27AE60 40%,
                                          #E67E22 40%, #E67E22 60%,
                                          #E74C3C 60%, #E74C3C 100%);
    border-radius: 8px; height: 14px; position: relative; overflow: visible;
  }
  .gauge-needle {
    position: absolute; top: -5px; width: 4px; height: 24px;
    background: #1F2937; border-radius: 2px;
    transform: translateX(-50%);
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }
  .gauge-labels {
    display: flex; justify-content: space-between;
    font-size: 0.72rem; color: #888; margin-top: 4px;
  }

  /* Metrické karty */
  .m-card {
    background: #F8FAFF; border: 1px solid #E0E8F5; border-radius: 10px;
    padding: 10px 14px; text-align: center;
  }
  .m-val   { font-size: 1.5rem; font-weight: 700; color: #0D47A1; }
  .m-label { font-size: 0.7rem; color: #888; text-transform: uppercase; margin-top: 2px; }

  /* Disclaimer */
  .disclaimer {
    background: #FFF8E1; border: 1px solid #FFD54F;
    border-radius: 10px; padding: 10px 16px;
    font-size: 0.83rem; color: #5D4037; margin-bottom: 16px;
  }

  /* Tlačidlo */
  .stButton > button {
    background: linear-gradient(135deg, #0D47A1, #1976D2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-size: 1rem !important;
    font-weight: 600 !important; padding: 14px 0 !important;
    width: 100% !important; margin-top: 8px !important;
    transition: all 0.2s !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 12px rgba(13,71,161,0.25) !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #0A3880, #1565C0) !important;
    box-shadow: 0 6px 16px rgba(13,71,161,0.35) !important;
    transform: translateY(-1px) !important;
  }

  /* Inputs */
  div[data-testid="stNumberInput"] input {
    font-size: 1rem !important; border-radius: 8px !important;
  }
  div[data-testid="stSelectbox"] > div { border-radius: 8px !important; }

  /* Radio – prepínače áno/nie */
  div[data-testid="stRadio"] label {
    font-size: 0.88rem !important;
  }
  div[data-testid="stRadio"] > div {
    gap: 6px !important;
  }

  /* Feature importance pruhy */
  .fi-row {
    display: flex; align-items: center; gap: 8px;
    margin: 5px 0; font-size: 0.8rem;
  }
  .fi-label { width: 180px; color: #444; flex-shrink: 0; }
  .fi-track { flex: 1; background: #EEF2F8; border-radius: 6px; height: 9px; }
  .fi-bar   {
    height: 9px; border-radius: 6px;
    background: linear-gradient(to right, #1565C0, #0D47A1);
  }
  .fi-val   { width: 40px; text-align: right; color: #666; }

  /* Info box */
  .info-box {
    background: #EBF3FB; border-left: 4px solid #1976D2;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #1a3a5c; margin: 6px 0;
  }

  /* Popis otázky */
  .q-help { font-size: 0.75rem; color: #888; margin-top: -4px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL – Random Forest (implementácia from scratch)
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ["Pohlavie_enc","Vek","TK_sys","TK_dia","Pulz",
            "E5","D2","K4","H1","K1","P9","E1","N1","P12","F9","E3"]

FEATURE_LABELS = {
    "Pohlavie_enc": "Pohlavie",
    "Vek":    "Vek",
    "TK_sys": "TK systolický",
    "TK_dia": "TK diastolický",
    "Pulz":   "Pulz",
    "E5":  "E5 – Synkopa v rodine",
    "D2":  "D2 – Strata vedomia pri vstávaní",
    "K4":  "K4 – Epizóda pri dlhom státí",
    "H1":  "H1 – Pocit tepla pred epizódou",
    "K1":  "K1 – Spúšťač: ortostáza",
    "P9":  "P9 – Úraz pri páde",
    "E1":  "E1 – Rod. anamnéza srdca",
    "N1":  "N1 – Predchádzajúca synkopa",
    "P12": "P12 – Trvanie bezvedomia",
    "F9":  "F9 – Nevoľnosť po epizóde",
    "E3":  "E3 – Únava po epizóde",
}

QUESTIONNAIRE = [
    ("E5",  "E5 – Mal niekto z rodiny synkopu alebo kolaps?"),
    ("D2",  "D2 – Strata vedomia pri vstávaní?"),
    ("K4",  "K4 – Epizóda nastala po dlhom státí?"),
    ("H1",  "H1 – Pacient mal pocit tepla pred synkopou?"),
    ("K1",  "K1 – Synkopu spustilo vstávanie (ortostáza)?"),
    ("P9",  "P9 – Pacient utrpel úraz pri páde?"),
    ("E1",  "E1 – Rodinná anamnéza srdcových ochorení?"),
    ("N1",  "N1 – Pacient mal synkopu v minulosti?"),
    ("P12", "P12 – Bezvedomie trvalo dlhšie ako obvykle?"),
    ("F9",  "F9 – Nevoľnosť po epizóde?"),
    ("E3",  "E3 – Únava po synkope?"),
]

RADIO_OPTS = ["—", "Nie", "Áno"]

def radio_to_val(v):
    if v == "Nie": return 0.0
    if v == "Áno": return 1.0
    return np.nan


# ── Decision Tree ──────────────────────────────────────────────────────────────
class DTree:
    def __init__(self, max_depth=12, min_split=4, max_features=None):
        self.max_depth    = max_depth
        self.min_split    = min_split
        self.max_features = max_features

    def _gini(self, y):
        p = y.mean()
        return 1 - p*p - (1-p)**2

    def _split(self, X, y):
        best = (-1, None, None, -1)
        n = len(y); pg = self._gini(y); nf = X.shape[1]
        fi_ = (np.random.choice(nf, self.max_features, replace=False)
               if self.max_features and self.max_features < nf else range(nf))
        for fi in fi_:
            vals = np.unique(X[:, fi])
            if len(vals) < 2: continue
            for t in (vals[:-1] + vals[1:]) / 2:
                lm = X[:, fi] <= t
                nl, nr = lm.sum(), (~lm).sum()
                if nl < 2 or nr < 2: continue
                g = pg - (nl/n)*self._gini(y[lm]) - (nr/n)*self._gini(y[~lm])
                if g > best[3]:
                    best = (fi, t, lm, g)
        return best

    def _build(self, X, y, d, n_total):
        if d >= self.max_depth or len(y) < self.min_split or self._gini(y) < 1e-7:
            return {'leaf': True, 'p': float(y.mean())}
        fi, t, lm, g = self._split(X, y)
        if fi == -1 or g < 1e-7:
            return {'leaf': True, 'p': float(y.mean())}
        self.fi_acc_[fi] = self.fi_acc_.get(fi, 0) + (len(y) / n_total) * g
        return {'leaf': False, 'fi': fi, 't': t,
                'L': self._build(X[lm],  y[lm],  d+1, n_total),
                'R': self._build(X[~lm], y[~lm], d+1, n_total)}

    def fit(self, X, y):
        self.fi_acc_ = {}
        self.tree_   = self._build(X, y.astype(float), 0, len(y))
        return self

    def _p1(self, x, n):
        if n['leaf']: return n['p']
        return self._p1(x, n['L'] if x[n['fi']] <= n['t'] else n['R'])

    def predict_proba(self, X):
        return np.array([self._p1(x, self.tree_) for x in X])


class RandomForest:
    def __init__(self, n=200, max_depth=12, seed=42):
        self.n = n; self.max_depth = max_depth; self.seed = seed
        self.trees_ = []; self.fidx_ = []

    def fit(self, X, y):
        np.random.seed(self.seed)
        ns, nf = X.shape
        mf = max(1, int(np.sqrt(nf)))
        for _ in range(self.n):
            bi = np.random.choice(ns, ns, replace=True)
            fi = np.random.choice(nf, mf, replace=False)
            tree = DTree(max_depth=self.max_depth, min_split=4, max_features=mf)
            tree.fit(X[np.ix_(bi, fi)], y[bi])
            self.trees_.append(tree)
            self.fidx_.append(fi)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X[:, fi])
                        for t, fi in zip(self.trees_, self.fidx_)], axis=0)

    def feature_importances(self, n_features):
        fi_global = np.zeros(n_features)
        for tree, fidx in zip(self.trees_, self.fidx_):
            for local_j, imp in tree.fi_acc_.items():
                global_j = fidx[local_j]
                fi_global[global_j] += imp
        total = fi_global.sum()
        if total > 0:
            fi_global /= total
        return fi_global


@st.cache_resource(show_spinner="Inicializujem model… (prvé spustenie, ~30 s)")
def get_trained_model():
    import os, pandas as pd

    base = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(base, 'data_full1.csv')

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return None, None, {}, {}

    def parse_bp(val):
        val = str(val).strip()
        if val in ("-1","","nan","NEMERAT","NEMER","NEMERST"): return np.nan, np.nan
        first = val.split("-")[0].split(",")[0].strip()
        if "/" in first:
            parts = first.split("/")
            try:
                s = float(parts[0])
                d = float(parts[1].strip()) if parts[1].strip() not in ("-","","nan") else np.nan
                return s, d
            except: return np.nan, np.nan
        return np.nan, np.nan

    bp = df["A2"].map(parse_bp)
    df["TK_sys"]       = [x[0] for x in bp]
    df["TK_dia"]       = [x[1] for x in bp]
    df["Pulz"]         = pd.to_numeric(df["A3"], errors="coerce").replace(-1, np.nan)
    df["Pohlavie_enc"] = (df["Pohlavie"] == "M").astype(float)

    META = {"Pohlavie","Pohlavie_enc","Vek","Synkopa","Typ Synkopy",
            "A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
            "TK_sys","TK_dia","Pulz","Dátum","Datum narodenia","S","Číslo dotazníka"}
    for col in df.columns:
        if col not in META:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-1, np.nan)

    y = df["Synkopa"].astype(int).values

    np.random.seed(42)
    idx0 = np.where(y == 0)[0]; np.random.shuffle(idx0)
    idx1 = np.where(y == 1)[0]; np.random.shuffle(idx1)
    n0 = max(1, int(len(idx0) * 0.2))
    n1 = max(1, int(len(idx1) * 0.2))
    te_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    tr_idx = np.concatenate([idx0[n0:],  idx1[n1:]])

    X_tr = df[FEATURES].iloc[tr_idx].values.astype(float)
    y_tr = y[tr_idx]
    X_te = df[FEATURES].iloc[te_idx].values.astype(float)
    y_te = y[te_idx]

    medians   = np.nanmedian(X_tr, axis=0)
    X_tr_imp  = np.where(np.isnan(X_tr), medians, X_tr)
    X_te_imp  = np.where(np.isnan(X_te), medians, X_te)

    model = RandomForest(n=200, max_depth=12, seed=42)
    model.fit(X_tr_imp, y_tr)

    fi_arr   = model.feature_importances(len(FEATURES))
    feat_imp = {FEATURES[i]: float(fi_arr[i]) for i in range(len(FEATURES))}

    probs_te = model.predict_proba(X_te_imp)
    preds_te = (probs_te >= 0.50).astype(int)
    TP = int(((preds_te == 1) & (y_te == 1)).sum())
    FP = int(((preds_te == 1) & (y_te == 0)).sum())
    FN = int(((preds_te == 0) & (y_te == 1)).sum())
    TN = int(((preds_te == 0) & (y_te == 0)).sum())

    return model, medians, feat_imp, {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'n': len(te_idx)}


def do_predict(inputs, model, medians):
    x     = np.array([inputs.get(f, np.nan) for f in FEATURES], dtype=float)
    x_imp = np.where(np.isnan(x), medians, x)
    return float(model.predict_proba(x_imp.reshape(1, -1))[0])


# ── LOGGING ────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH  = os.path.join(_BASE_DIR, "predikcie_log.csv")

LOG_COLUMNS = [
    "id", "cas", "pravdepodobnost_%", "riziko", "skutocny_vysledok",
    "Pohlavie", "Vek", "TK_sys", "TK_dia", "Pulz",
    "E5", "D2", "K4", "H1", "K1", "P9", "E1", "N1", "P12", "F9", "E3"
]
# Index stĺpca skutocny_vysledok (1-based pre gspread)
_VYSLEDOK_COL = LOG_COLUMNS.index("skutocny_vysledok") + 1

def _fmt(v):
    """Formátuje hodnotu pre log – NaN → prázdny reťazec."""
    if v is None: return ""
    try:
        return "" if (isinstance(v, float) and np.isnan(v)) else v
    except: return v

def _build_row(session_id, inputs, prob, risk_label, pohlavie_str, skutocny_vysledok=""):
    return [
        session_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        round(prob * 100, 1),
        risk_label,
        skutocny_vysledok,
        pohlavie_str,
        _fmt(inputs.get("Vek")),
        _fmt(inputs.get("TK_sys")),
        _fmt(inputs.get("TK_dia")),
        _fmt(inputs.get("Pulz")),
        _fmt(inputs.get("E5")),  _fmt(inputs.get("D2")),
        _fmt(inputs.get("K4")),  _fmt(inputs.get("H1")),
        _fmt(inputs.get("K1")),  _fmt(inputs.get("P9")),
        _fmt(inputs.get("E1")),  _fmt(inputs.get("N1")),
        _fmt(inputs.get("P12")), _fmt(inputs.get("F9")),
        _fmt(inputs.get("E3")),
    ]

# ── Google Sheets pripojenie (cachované) ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_gsheet():
    """Vráti worksheet Google Sheets. Vyžaduje secrets v Streamlit Cloud."""
    if not GSPREAD_AVAILABLE:
        return None
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        sheet  = client.open_by_key(st.secrets["gsheet_id"]).sheet1
        # Hlavička – ak je list prázdny
        if not sheet.get_all_values():
            sheet.append_row(LOG_COLUMNS)
        return sheet
    except Exception:
        return None

def log_prediction(inputs, prob, risk_label, pohlavie_str):
    """
    Uloží predikciu do Google Sheets + CSV zálohy.
    Vracia session_id (pre neskoršie doplnenie skutočného výsledku).
    """
    session_id = str(uuid.uuid4())[:8]
    row = _build_row(session_id, inputs, prob, risk_label, pohlavie_str)

    # ── Google Sheets ─────────────────────────────────────────────────────────
    sheet = _get_gsheet()
    if sheet is not None:
        try:
            sheet.append_row(row, value_input_option="USER_ENTERED")
        except Exception:
            pass

    # ── CSV záloha ────────────────────────────────────────────────────────────
    try:
        file_exists = os.path.isfile(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(LOG_COLUMNS)
            writer.writerow(row)
    except Exception:
        pass

    return session_id

def update_skutocny_vysledok(session_id, vysledok):
    """
    Nájde riadok podľa session_id a doplní skutočný výsledok HUTT testu.
    Funguje pre Google Sheets aj CSV zálohu.
    """
    ok_sheet = False
    # ── Google Sheets ─────────────────────────────────────────────────────────
    sheet = _get_gsheet()
    if sheet is not None:
        try:
            cell = sheet.find(session_id, in_column=1)
            if cell:
                sheet.update_cell(cell.row, _VYSLEDOK_COL, vysledok)
                ok_sheet = True
        except Exception:
            pass

    # ── CSV záloha ────────────────────────────────────────────────────────────
    try:
        if os.path.isfile(LOG_PATH):
            rows = []
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for r in reader:
                    if r and r[0] == session_id:
                        r[_VYSLEDOK_COL - 1] = vysledok
                    rows.append(r)
            with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
    except Exception:
        pass

    return ok_sheet


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── HLAVIČKA ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div style="font-size:2.4rem;">🫀</div>
  <div>
    <h1>Predikcia výsledku HUTT testu</h1>
    <div class="subtitle">Rozhodovacia podpora pre klinického lekára · Head-Up Tilt Test</div>
    <span class="badge">Random Forest · AUC 80.8 %</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Len rozhodovacia podpora.</strong>
  Finálne klinické rozhodnutie je vždy na lekárovi. Výsledok modelu nenahrádza klinické vyšetrenie.
</div>
""", unsafe_allow_html=True)

# Načítanie modelu
model, medians, feat_imp, cm = get_trained_model()
if model is None:
    st.error("❌ Model sa nenačítal – skontrolujte, či je súbor `data_full1.csv` v rovnakom priečinku ako `app.py`.")
    st.stop()

# ── FORMULÁR + VÝSLEDOK ───────────────────────────────────────────────────────
col_form, col_result = st.columns([1.15, 0.85], gap="large")

with col_form:
    # ── Anamnestické údaje ────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📋 Anamnestické údaje</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        pohlavie     = st.selectbox("Pohlavie", ["Žena", "Muž"], label_visibility="visible")
        pohlavie_enc = 1.0 if pohlavie == "Muž" else 0.0
    with c2:
        vek  = st.number_input("Vek (roky)", min_value=10, max_value=100, value=50, step=1)
    with c3:
        pulz = st.number_input("Pulz (bpm)", min_value=30, max_value=200, value=72, step=1)

    c4, c5 = st.columns(2)
    with c4:
        tk_sys = st.number_input("TK systolický (mmHg)", min_value=60, max_value=250, value=120, step=1)
    with c5:
        tk_dia = st.number_input("TK diastolický (mmHg)", min_value=40, max_value=150, value=75, step=1)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Dotazník – prepínače Áno / Nie ───────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📝 Dotazník pacienta</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
Vyberte <strong>Áno</strong> alebo <strong>Nie</strong> pre každú otázku.
Ak odpoveď nie je známa, nechajte <strong>—</strong> (systém doplní typickú hodnotu z datasetu).
</div>
""", unsafe_allow_html=True)

    st.write("")  # malý priestor

    dotaz_vals = {}
    # Zobrazíme v 2 stĺpcoch pre úsporu miesta
    q_left  = QUESTIONNAIRE[:6]
    q_right = QUESTIONNAIRE[6:]

    qa, qb = st.columns(2)
    with qa:
        for key, label in q_left:
            v = st.radio(label, RADIO_OPTS, horizontal=True, key=key, index=0)
            dotaz_vals[key] = radio_to_val(v)
    with qb:
        for key, label in q_right:
            v = st.radio(label, RADIO_OPTS, horizontal=True, key=key, index=0)
            dotaz_vals[key] = radio_to_val(v)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍  Predikovať výsledok HUTT testu", use_container_width=True)


# ── VÝSLEDKY ──────────────────────────────────────────────────────────────────
with col_result:

    if predict_btn:
        inputs = {
            "Pohlavie_enc": pohlavie_enc,
            "Vek":   float(vek),
            "TK_sys": float(tk_sys),
            "TK_dia": float(tk_dia),
            "Pulz":   float(pulz),
            **dotaz_vals
        }

        prob = do_predict(inputs, model, medians)
        pct  = prob * 100
        needle_pct = min(max(int(pct), 0), 100)

        # Risk level
        if prob >= 0.60:
            rc = "result-high";   rl = "🔴 Vysoké riziko synkopy"; rl_plain = "Vysoké"
            rd = "Model predikuje <strong>pozitívny</strong> výsledok HUTT testu."
            rr_cls = "rec-high"
            rec  = "💊 <strong>Odporúčanie:</strong> Zvážte priame indikácie HUTT testu. " \
                   "Výsledok naznačuje vasovagálnu synkopu – odporúča sa ďalšie kardiologické sledovanie."
        elif prob >= 0.40:
            rc = "result-medium"; rl = "🟡 Stredné riziko synkopy"; rl_plain = "Stredné"
            rd = "Výsledok je <strong>neistý</strong> – odporúča sa klinické zváženie."
            rr_cls = "rec-medium"
            rec  = "🩺 <strong>Odporúčanie:</strong> Doplňte klinické vyšetrenie. " \
                   "Výsledok je hraničný, rozhodnutie závisí od ďalšieho kontextu pacienta."
        else:
            rc = "result-low";    rl = "🟢 Nízke riziko synkopy"; rl_plain = "Nízke"
            rd = "Model predikuje <strong>negatívny</strong> výsledok HUTT testu."
            rr_cls = "rec-low"
            rec  = "✅ <strong>Odporúčanie:</strong> Nízka pravdepodobnosť synkopy. " \
                   "Zvážte alternatívne príčiny straty vedomia."

        # ── Záznam do logu ────────────────────────────────────────────────────
        try:
            sid = log_prediction(inputs, prob, rl_plain, pohlavie)
            st.session_state["last_sid"] = sid
            st.caption("💾 Predikcia zaznamenaná do logu.")
        except Exception as log_err:
            st.session_state["last_sid"] = None
            st.caption(f"⚠️ Log sa neuložil: {log_err}")

        # Výsledkový box
        st.markdown(f"""
<div class="result-box {rc}">
  <div class="result-prob">{pct:.1f}&nbsp;%</div>
  <div class="result-label">{rl}</div>
  <div class="result-desc">{rd}</div>
</div>
""", unsafe_allow_html=True)

        # Gauge lišta s ihlou
        st.markdown(f"""
<div class="gauge-wrap">
  <div class="gauge-track">
    <div class="gauge-needle" style="left:{needle_pct}%;"></div>
  </div>
  <div class="gauge-labels">
    <span>0 % – Nízke</span>
    <span>40 %</span>
    <span>60 %</span>
    <span>Vysoké – 100 %</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # Klinické odporúčanie
        st.markdown(f'<div class="rec-box {rr_cls}">{rec}</div>', unsafe_allow_html=True)

        # ── Výsledok HUTT testu (doplniť po vyšetrení) ───────────────────────
        st.markdown("""
<div style="background:#F8FAFF; border:1.5px dashed #90B8E8; border-radius:10px;
            padding:14px 18px; margin:10px 0 6px 0;">
  <div style="font-size:0.82rem; font-weight:700; color:#0D47A1;
              text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">
    ✏️ Výsledok HUTT testu (vyplňte po vyšetrení)
  </div>
  <div style="font-size:0.82rem; color:#666; margin-bottom:6px;">
    Vyberte skutočný výsledok a stlačte <strong>Zapísať</strong> –
    doplní sa do záznamu tohto pacienta.
  </div>
</div>
""", unsafe_allow_html=True)

        vysledok_opts = ["— nevyplnené —", "Pozitívny (synkopa nastala)",
                         "Negatívny (synkopa nenastala)"]
        vysledok_sel = st.radio(
            "Skutočný výsledok HUTT testu:",
            vysledok_opts,
            horizontal=True,
            key="vysledok_radio",
            label_visibility="collapsed"
        )
        if st.button("✅  Zapísať výsledok do logu", key="btn_vysledok",
                     disabled=(vysledok_sel == "— nevyplnené —")):
            sid = st.session_state.get("last_sid")
            if sid:
                val = "Pozitívny" if "Pozitívny" in vysledok_sel else "Negatívny"
                update_skutocny_vysledok(sid, val)
                st.success(f"✅ Výsledok **{val}** bol zapísaný do záznamu.")
            else:
                st.warning("⚠️ ID záznamu sa nenašlo – výsledok sa nedal doplniť.")

        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

        # ── Výkonnostné metriky ───────────────────────────────────────────────
        sens_val = (cm['TP'] / (cm['TP'] + cm['FN']) * 100) if (cm['TP'] + cm['FN']) > 0 else 0.0
        spec_val = (cm['TN'] / (cm['TN'] + cm['FP']) * 100) if (cm['TN'] + cm['FP']) > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="m-card"><div class="m-val">80.8%</div><div class="m-label">AUC (CV)</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="m-card"><div class="m-val">{sens_val:.1f}%</div><div class="m-label">Senzitivita</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="m-card"><div class="m-val">{spec_val:.1f}%</div><div class="m-label">Špecificita</div></div>', unsafe_allow_html=True)

        # ── Detaily modelu v expanderi ────────────────────────────────────────
        with st.expander("📈 Detaily modelu (pre výskum)"):
            st.markdown("**Dôležitosť atribútov (top 8)**")
            if feat_imp and max(feat_imp.values()) > 0:
                top_fi = sorted(feat_imp.items(), key=lambda x: -x[1])[:8]
                max_fi = top_fi[0][1]
                for feat, imp in top_fi:
                    label = FEATURE_LABELS.get(feat, feat)
                    bar_w = int(imp / max_fi * 100) if max_fi > 0 else 0
                    st.markdown(
                        f"<div class='fi-row'>"
                        f"<span class='fi-label'>{label}</span>"
                        f"<div class='fi-track'><div class='fi-bar' style='width:{bar_w}%;'></div></div>"
                        f"<span class='fi-val'>{imp:.3f}</span>"
                        f"</div>", unsafe_allow_html=True)

            n_te = cm.get('n', cm['TP'] + cm['FP'] + cm['FN'] + cm['TN'])
            st.markdown(f"\n**Konfúzna matica** (testovacia sada, n={n_te})")
            st.markdown(f"""
| | Predik. synkopa | Predik. bez synkopy |
|---|:---:|:---:|
| **Skutočná synkopa** | TP = {cm['TP']} | FN = {cm['FN']} |
| **Bez synkopy** | FP = {cm['FP']} | TN = {cm['TN']} |
""")
            n_pos = cm['TP'] + cm['FN']
            st.markdown(f"""
<div class="info-box">
Model prehliadne <strong>{cm['FN']}</strong> zo <strong>{n_pos}</strong> skutočných synkop (FN).
Falošných poplachov: <strong>{cm['FP']}</strong>.
</div>""", unsafe_allow_html=True)

    else:
        # Prázdny stav – návod
        st.markdown("""
<div class="card" style="text-align:center; padding: 36px 22px;">
  <div style="font-size:3rem; margin-bottom:12px;">🩺</div>
  <div style="font-size:1.05rem; font-weight:600; color:#0D47A1; margin-bottom:8px;">
    Ako začať
  </div>
  <div style="font-size:0.9rem; color:#555; line-height:1.7;">
    1. Zadajte <strong>anamnestické údaje</strong> pacienta<br>
    2. Označte odpovede z <strong>dotazníka</strong> (Áno / Nie)<br>
    3. Stlačte <strong>Predikovať</strong><br>
    4. Prečítajte <strong>výsledok</strong> a odporúčanie
  </div>
</div>
""", unsafe_allow_html=True)

        # Informácie o modeli
        st.markdown("""
<div class="card">
  <div class="card-title">ℹ️ O modeli</div>
  <div style="font-size:0.88rem; color:#444; line-height:1.7;">
    <strong>Algoritmus:</strong> Random Forest (200 stromov)<br>
    <strong>Dataset:</strong> 371 pacientov · HUTT test<br>
    <strong>Tréning:</strong> 5-fold krížová validácia<br>
    <strong>Cieľová premenná:</strong> Synkopa (0 = nie, 1 = áno)<br>
    <strong>Vstupné dáta:</strong> Anamnéza + dotazník (16 atribútov)
  </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #DDE3EE;margin:20px 0 10px 0;">
<div style="text-align:center;font-size:0.78rem;color:#AAB;">
  Predikcia synkopy · Random Forest · AUC CV 80.8 % · Senzitivita 92.7 % ·
  <em>Len pre výskumné a edukačné účely</em>
</div>
""", unsafe_allow_html=True)
