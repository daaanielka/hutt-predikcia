"""
Detailná validácia — hutt_analysis
=======================================
Bakalárska práca: Predikcia výsledku HUTT testu

Spúšťa sa jednorazovo po výbere finálnych modelov z analyza.py.
Predspracovanie dát je identické s analyza.py.

Obsah:
  1. OOF predikcie pre vybraté modely (5-fold CV)
  2. Bootstrap CI pre AUC (1000 iterácií, stratifikovaný)
  3. Kalibračné krivky + Brier score (s izotonou regresiou)
  4. Optimálny prah — Youden's J index + PPV/NPV + 95% CI
  5. SHAP hodnoty — P3 Extra Trees (TreeExplainer)
  6. Decision Curve Analysis (DCA)
  7. Permutačný test (n=1000)
  8. Fairness analýza — Pohlavie a Vek subskupiny

Výstup:
  validacia.xlsx  (sheets: kalibracia, prah_youden, shap, permutacny_test, fairness)
  validacia_consort.png / _kalibracia.png / _shap_bar.png / _dca.png

Vybraté modely (top z analyza.py):
  - P1  Extra Trees   (anamnéza, 5 features, AUC=0.800)
  - P3  Extra Trees   (kombinácia, 13 features, AUC=0.800)
  - P4  Stacking      (ET+KNN+LR→Ridge na P3, AUC=0.796)

Spustenie:
    python validacia.py
"""

import math
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model_components import ConsensusFeatureSelector, P3SelectorConsensus
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              confusion_matrix, f1_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin, clone

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("UPOZORNENIE: shap nie je nainštalovaný — sekcia 5 bude preskočená.")
    print("  pip install shap")

import joblib

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
N_BOOTSTRAP = 1000

# True  = P1–P7 vypadnú, vznikne Ma_diag_srdcove_ochorenie
# False = P1–P7 ostanú ako samostatné features (bez kompozitu)
USE_MA_DIAG = False

_val_sheets: dict = {}   # collector — každý df sa uloží tu, na konci → validacia.xlsx


def make_pipe(clf, apply_fs=False, scale=False, n_p1=0):
    steps = [('imp', SimpleImputer(strategy='median'))]
    if apply_fs:
        if n_p1 > 0:
            steps.append(('fs', P3SelectorConsensus(n_p1=n_p1)))
        else:
            steps.append(('fs', ConsensusFeatureSelector()))
    if scale:
        steps.append(('scaler', RobustScaler()))
    steps.append(('clf', clf))
    return Pipeline(steps)


# =============================================================
# PREDSPRACOVANIE  (identické s analyza.py)
# =============================================================

print("=" * 65)
print("NAČÍTANIE A PREDSPRACOVANIE DÁT")
print("=" * 65)

df = pd.read_csv('data_full1.csv')
_n_raw = len(df)
_n_no_a10 = int(
    (df['A10'].isna() | (df['A10'] == -1)).sum()
) if 'A10' in df.columns else 0

admin_cols         = ['Číslo dotazníka', 'Dátum', 'Datum narodenia']
leakage_cols       = ['Synkopa', 'Typ Synkopy']
post_test_cols     = ['A1', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
flag_cols          = ['A']
poor_quality_cols  = ['R', 'R3', 'C']
text_cols          = ['S']
doctor_remove_cols = ['B2', 'C3', 'J1', 'N7', 'P32',
                      'Q1', 'Q4', 'Q12', 'Q13', 'Q16', 'Q17', 'Q18']
p_srdc_cols = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
if USE_MA_DIAG:
    ALL_REMOVE = (admin_cols + leakage_cols + post_test_cols +
                  flag_cols + poor_quality_cols + text_cols +
                  doctor_remove_cols + p_srdc_cols)
else:
    ALL_REMOVE = (admin_cols + leakage_cols + post_test_cols +
                  flag_cols + poor_quality_cols + text_cols +
                  doctor_remove_cols)

df_work = df.copy()
int_cols   = df_work.select_dtypes(include='int64').columns.tolist()
float_cols = df_work.select_dtypes(include='float64').columns.tolist()
df_work[int_cols]   = df_work[int_cols].replace(-1, np.nan)
df_work[float_cols] = df_work[float_cols].replace(-1, np.nan)
for col in ['A2', 'B1', 'J3']:
    if col in df_work.columns:
        df_work[col] = df_work[col].replace(
            {'-1': np.nan, '-': np.nan, 'Ч': np.nan, 'NEMERAT': np.nan})

p_srdc = [c for c in p_srdc_cols if c in df_work.columns]
if USE_MA_DIAG:
    df_work['Ma_diag_srdcove_ochorenie'] = df_work[p_srdc].max(axis=1)
df_work = df_work.drop(columns=ALL_REMOVE, errors='ignore')

if 'A2' in df_work.columns:
    bp = df_work['A2'].str.split('/', expand=True)
    df_work['A2_sys'] = pd.to_numeric(bp[0], errors='coerce')
    df_work['A2_dia'] = pd.to_numeric(
        bp[1] if bp.shape[1] > 1 else pd.Series([np.nan] * len(df_work)),
        errors='coerce')
    df_work.drop(columns=['A2'], inplace=True)

df_work['Pohlavie'] = df_work['Pohlavie'].map({'F': 0, 'M': 1})
for col in ['B1', 'J3']:
    if col in df_work.columns:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

df_work = df_work.dropna(subset=['A10'])
df_work['A10'] = df_work['A10'].astype(int)
y       = df_work['A10'].values
df_feat = df_work.drop(columns=['A10']).copy()

P1_COLS = [c for c in ['A2_sys', 'A2_dia', 'A3', 'Vek', 'Pohlavie']
           if c in df_feat.columns]
a_cols   = [c for c in df_feat.columns if c.startswith('A')]
dem_cols = ['Vek', 'Pohlavie']
P2_POOL  = [c for c in df_feat.columns if c not in a_cols + dem_cols]
P3_POOL  = P1_COLS + [c for c in P2_POOL if c not in P1_COLS]
N_P1     = len(P1_COLS)

print("=" * 65)
print("OPIS DATASETU")
print("=" * 65)
_n_total   = len(y)
_n_pos     = int(y.sum())
_n_neg     = int((1 - y).sum())
_prev      = _n_pos / _n_total * 100
print(f"  Celkový počet pacientov : {_n_total}")
print(f"  HUTT pozitívny (A10=1)  : {_n_pos}  ({_prev:.1f} %)")
print(f"  HUTT negatívny (A10=0)  : {_n_neg}  ({100-_prev:.1f} %)")
print(f"  Pomer tried (neg/pos)   : {_n_neg/_n_pos:.2f}")

if 'Vek' in df_feat.columns:
    _vek = df_feat['Vek'].dropna()
    print(f"  Vek: priemer={_vek.mean():.1f}  "
          f"SD={_vek.std():.1f}  "
          f"medián={_vek.median():.0f}  "
          f"rozsah=[{_vek.min():.0f}–{_vek.max():.0f}]")

if 'Pohlavie' in df_feat.columns:
    _poh = df_feat['Pohlavie'].dropna()
    _n_f = int((_poh == 0).sum())
    _n_m = int((_poh == 1).sum())
    print(f"  Pohlavie: F={_n_f} ({_n_f/_n_total*100:.1f} %)  "
          f"M={_n_m} ({_n_m/_n_total*100:.1f} %)")
    # prevalencia A10=1 per pohlavie
    _idx_f = df_feat.index[df_feat['Pohlavie'] == 0]
    _idx_m = df_feat.index[df_feat['Pohlavie'] == 1]
    _pos_f = y[df_feat.index.get_indexer(_idx_f)]
    _pos_m = y[df_feat.index.get_indexer(_idx_m)]
    print(f"  Prevalencia HUTT+: F={_pos_f.mean()*100:.1f} %  "
          f"M={_pos_m.mean()*100:.1f} %")

print(f"  Chýbajúce hodnoty: {df_feat.isnull().any(axis=1).sum()} pacientov "
      f"má aspoň 1 chýbajúcu hodnotu (imputácia mediánom)")
print(f"  Feature pooly: P1={len(P1_COLS)}  P2 pool={len(P2_POOL)}  "
      f"P3 pool={len(P3_POOL)}")
print()


# =============================================================
# CONSORT FLOWCHART
# =============================================================

_n_analyzed = len(y)
_n_excluded = _n_raw - _n_analyzed

fig_cs, ax_cs = plt.subplots(figsize=(7, 6))
ax_cs.set_xlim(0, 10)
ax_cs.set_ylim(0, 10)
ax_cs.axis('off')
fig_cs.patch.set_facecolor('white')

def _box(ax, x, y_pos, w, h, text, color='#DDEEFF'):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y_pos - h/2), w, h,
        boxstyle='round,pad=0.15', linewidth=1.2,
        edgecolor='#334466', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y_pos, text, ha='center', va='center',
            fontsize=9, wrap=True,
            multialignment='center')

def _arrow(ax, x, y_top, y_bot):
    ax.annotate('', xy=(x, y_bot + 0.05), xytext=(x, y_top - 0.05),
                arrowprops=dict(arrowstyle='->', color='#334466', lw=1.5))

_box(ax_cs, 5, 8.8, 6.5, 1.2,
     f"Všetky záznamy v databáze\nn = {_n_raw}", color='#E8F0FB')

_arrow(ax_cs, 5, 8.2, 7.2)

if _n_excluded > 0:
    _box(ax_cs, 8.2, 7.35, 3.2, 0.9,
         f"Vylúčení\n(chýbajúca A10)\nn = {_n_excluded}", color='#FFF0F0')
    ax_cs.annotate('', xy=(6.6, 7.35), xytext=(5, 7.6),
                   arrowprops=dict(arrowstyle='->', color='#334466', lw=1.2))

_box(ax_cs, 5, 6.5, 6.5, 1.2,
     f"Analyzovaní pacienti\nn = {_n_analyzed}", color='#E8F0FB')

_arrow(ax_cs, 5, 5.9, 4.9)

_box(ax_cs, 5, 4.3, 6.5, 1.6,
     f"5-fold stratifikovaná CV\n"
     f"HUTT pozitívny (A10=1): n = {int(y.sum())}  "
     f"({y.mean()*100:.1f} %)\n"
     f"HUTT negatívny (A10=0): n = {int((1-y).sum())}  "
     f"({(1-y).mean()*100:.1f} %)", color='#E8F0FB')

_arrow(ax_cs, 5, 3.5, 2.7)

_box(ax_cs, 5, 2.1, 6.5, 1.1,
     "Vyhodnotenie: OOF AUC, Bootstrap CI,\n"
     "Kalibrácia, DCA, Permutačný test, Fairness", color='#EEF8EE')

ax_cs.set_title('CONSORT diagram — tok pacientov', fontsize=11, pad=8)
plt.tight_layout()
plt.savefig('validacia_consort.png', dpi=150, bbox_inches='tight')
plt.close()
print("Uložené: validacia_consort.png")


# =============================================================
# VYBRATÉ MODELY
# =============================================================

pos_weight = float((y == 0).sum()) / float((y == 1).sum())

SELECTED = {
    'P1_ET': {
        'label': 'P1 · Extra Trees',
        'X':     df_feat[P1_COLS],
        'pipe':  make_pipe(ExtraTreesClassifier(
                     n_estimators=200, random_state=RANDOM_STATE,
                     n_jobs=1, class_weight='balanced'),
                     apply_fs=False),
        'n_p1':  0,
    },
    'P3_ET': {
        'label': 'P3 · Extra Trees',
        'X':     df_feat[P3_POOL],
        'pipe':  make_pipe(ExtraTreesClassifier(
                     n_estimators=200, random_state=RANDOM_STATE,
                     n_jobs=1, class_weight='balanced'),
                     apply_fs=True, n_p1=N_P1),
        'n_p1':  N_P1,
    },
}

_STACK_BASE_NAMES = ['Extra Trees', 'KNN', 'Logist. regresia']
_STACK_CLFS = {
    'Extra Trees':      (ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE,
                             n_jobs=1, class_weight='balanced'), False),
    'KNN':              (KNeighborsClassifier(n_neighbors=7, weights='distance',
                             metric='euclidean'), True),
    'Logist. regresia': (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                             class_weight='balanced', C=0.1), True),
}


# =============================================================
# 1. OOF PREDIKCIE
# =============================================================

print("\n" + "=" * 65)
print("1. OOF PREDIKCIE (5-fold CV)")
print("=" * 65)

oof_probas = {}

for key, cfg in SELECTED.items():
    X_m   = cfg['X']
    oof   = np.zeros(len(y))
    for tr_idx, te_idx in CV.split(X_m, y):
        pipe = clone(cfg['pipe'])
        pipe.fit(X_m.iloc[tr_idx], y[tr_idx])
        oof[te_idx] = pipe.predict_proba(X_m.iloc[te_idx])[:, 1]
    oof_probas[key] = oof
    print(f"  {cfg['label']:<25}  OOF AUC = {roc_auc_score(y, oof):.4f}")

# P4 Stacking OOF
print(f"  {'P4 · Stacking':<25}  ", end='')
X_p3_s  = df_feat[P3_POOL]
_n_base = len(_STACK_BASE_NAMES)
oof_stack = np.zeros(len(y))
for tr_idx, te_idx in CV.split(X_p3_s, y):
    X_tr, X_te = X_p3_s.iloc[tr_idx], X_p3_s.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    meta_tr = np.zeros((len(tr_idx), _n_base))
    for in_tr, in_val in INNER_CV.split(X_tr, y_tr):
        for bi, bname in enumerate(_STACK_BASE_NAMES):
            clf_b, scale_b = _STACK_CLFS[bname]
            p = make_pipe(clone(clf_b), apply_fs=True, scale=scale_b, n_p1=N_P1)
            p.fit(X_tr.iloc[in_tr], y_tr[in_tr])
            meta_tr[in_val, bi] = p.predict_proba(X_tr.iloc[in_val])[:, 1]
    meta_clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                   max_iter=500, random_state=RANDOM_STATE)
    meta_clf.fit(meta_tr, y_tr)
    meta_te = np.zeros((len(te_idx), _n_base))
    for bi, bname in enumerate(_STACK_BASE_NAMES):
        clf_b, scale_b = _STACK_CLFS[bname]
        p = make_pipe(clone(clf_b), apply_fs=True, scale=scale_b, n_p1=N_P1)
        p.fit(X_tr, y_tr)
        meta_te[:, bi] = p.predict_proba(X_te)[:, 1]
    oof_stack[te_idx] = meta_clf.predict_proba(meta_te)[:, 1]

oof_probas['P4_stack'] = oof_stack
print(f"OOF AUC = {roc_auc_score(y, oof_stack):.4f}")

MODEL_LABELS = {
    'P1_ET':    'P1 · Extra Trees',
    'P3_ET':    'P3 · Extra Trees',
    'P4_stack': 'P4 · Stacking',
}
MODEL_COLORS = {
    'P1_ET':    'royalblue',
    'P3_ET':    'seagreen',
    'P4_stack': '#C357A5',
}


# =============================================================
# 2. BOOTSTRAP CI PRE AUC
# =============================================================

print("\n" + "=" * 65)
print("2. BOOTSTRAP CI PRE AUC  (n=1000, stratifikovaný)")
print("=" * 65)
print("  CI zachytáva neistotu zo vzorky pacientov pri fixných OOF predikciách.")
print("  Neistota trénovania modelu (výber foldov, náhodnosť) NIE je zahrnutá.")
print("  Model                     AUC     95% CI")
print("  " + "─" * 50)

boot_results = {}
rng = np.random.default_rng(RANDOM_STATE)

for key, oof in oof_probas.items():
    aucs = []
    for _ in range(N_BOOTSTRAP):
        # stratifikovaný bootstrap — zachová pomer tried
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        b0   = rng.choice(idx0, size=len(idx0), replace=True)
        b1   = rng.choice(idx1, size=len(idx1), replace=True)
        b    = np.concatenate([b0, b1])
        try:
            aucs.append(roc_auc_score(y[b], oof[b]))
        except ValueError:
            pass
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    boot_results[key] = (np.mean(aucs), lo, hi)
    print(f"  {MODEL_LABELS[key]:<25} {roc_auc_score(y, oof):.4f}  "
          f"[{lo:.4f} – {hi:.4f}]")


# =============================================================
# 3. KALIBRÁCIA + BRIER SCORE
# =============================================================

print("\n" + "=" * 65)
print("3. KALIBRÁCIA + BRIER SCORE")
print("=" * 65)

fig_cal, axes_cal = plt.subplots(1, 3, figsize=(15, 5))
fig_cal.suptitle('Kalibračné krivky — ConsensusFS (prahová logika)', fontsize=12)

_calib_grid = np.linspace(0.0, 1.0, 100)   # fixná os x pre interpoláciu bootstrap

calib_rows = []
for ax, (key, oof) in zip(axes_cal, oof_probas.items()):
    brier = brier_score_loss(y, oof)
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=5, strategy='quantile')

    # bootstrap CI pre kalibračnú krivku
    _cal_boot = []
    for _ in range(N_BOOTSTRAP):
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        b0   = rng.choice(idx0, size=len(idx0), replace=True)
        b1   = rng.choice(idx1, size=len(idx1), replace=True)
        b    = np.concatenate([b0, b1])
        try:
            _fp_b, _mp_b = calibration_curve(
                y[b], oof[b], n_bins=5, strategy='quantile')
            _cal_boot.append(np.interp(_calib_grid, _mp_b, _fp_b,
                                       left=np.nan, right=np.nan))
        except ValueError:
            pass
    _cal_boot = np.array(_cal_boot)
    _cal_lo   = np.nanpercentile(_cal_boot, 2.5, axis=0)
    _cal_hi   = np.nanpercentile(_cal_boot, 97.5, axis=0)
    _valid    = ~(np.isnan(_cal_lo) | np.isnan(_cal_hi))

    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Ideálna kalibrácia')
    ax.fill_between(_calib_grid[_valid], _cal_lo[_valid], _cal_hi[_valid],
                    alpha=0.20, color=MODEL_COLORS[key], label='95% CI (bootstrap)')
    ax.plot(mean_pred, frac_pos, 'o-', color=MODEL_COLORS[key], lw=2,
            label=f'Model\nBrier={brier:.4f}')
    ax.set_title(MODEL_LABELS[key], fontsize=10)
    ax.set_xlabel('Predikovaná pravdepodobnosť')
    ax.set_ylabel('Skutočná frekvencia')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    kvalita = ("dobrá" if brier < 0.20
               else "priemerná" if brier < 0.25
               else "slabá")
    print(f"  {MODEL_LABELS[key]:<25}  Brier = {brier:.4f}  ({kvalita})")
    calib_rows.append({'Model': MODEL_LABELS[key], 'Brier_score': round(brier, 4),
                       'Kvalita': kvalita, 'Baseline_Brier': 0.244})

plt.tight_layout()
plt.savefig('validacia_kalibracia.png', dpi=150, bbox_inches='tight')
plt.close()
print("Uložené: validacia_kalibracia.png")
_val_sheets['kalibracia'] = pd.DataFrame(calib_rows)
print("Pripravené: sheet 'kalibracia'")
_baseline_brier = (y == 1).mean() * (1 - (y == 1).mean())
print(f"\n  Referenčné hodnoty Brier skóre:")
print(f"    Ideálny model          : 0.000  (dokonalá predikcia)")
print(f"    Baseline (prevalencia) : {_baseline_brier:.3f}  (model vždy predikuje {y.mean()*100:.0f}%)")
print(f"    Horný limit            : 1.000  (najhorší možný)")
print(f"    → Model musí mať Brier < {_baseline_brier:.3f} aby prekonával náhodu.")


# =============================================================
# 4. OPTIMÁLNY PRAH — YOUDEN'S J INDEX + PPV/NPV + 95% CI
# =============================================================

print("\n" + "=" * 65)
print("4. OPTIMÁLNY PRAH — YOUDEN'S J INDEX + PPV/NPV + 95% CI")
print("=" * 65)
print("  Youden's J = Senzitivita + Specificita − 1  (maximalizovaný)")
print("  CI (bootstrap, n=1000, stratifikovaný) — neistota vzorky pacientov.")
print()

thresh_rows = []
for key, oof in oof_probas.items():
    fpr, tpr, thresholds = roc_curve(y, oof)
    spec_curve = 1 - fpr
    j_curve    = tpr + spec_curve - 1
    best_idx   = np.argmax(j_curve)
    thr_opt    = thresholds[best_idx]
    j_opt      = j_curve[best_idx]

    pred_opt = (oof >= thr_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_opt, labels=[0, 1]).ravel()
    sens_opt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_opt = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv_opt  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv_opt  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_opt   = f1_score(y, pred_opt, zero_division=0)

    # bootstrap CI pre metriky pri Youden prahu
    b_sens, b_spec, b_ppv, b_npv = [], [], [], []
    for _ in range(N_BOOTSTRAP):
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        b0   = rng.choice(idx0, size=len(idx0), replace=True)
        b1   = rng.choice(idx1, size=len(idx1), replace=True)
        b    = np.concatenate([b0, b1])
        yb   = y[b]
        pb   = (oof[b] >= thr_opt).astype(int)
        try:
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
            b_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0)
            b_spec.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)
            b_ppv.append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0)
            b_npv.append(tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0.0)
        except ValueError:
            pass

    ci_sens = np.percentile(b_sens, [2.5, 97.5])
    ci_spec = np.percentile(b_spec, [2.5, 97.5])
    ci_ppv  = np.percentile(b_ppv,  [2.5, 97.5])
    ci_npv  = np.percentile(b_npv,  [2.5, 97.5])

    # pri prahu 0.5
    pred_05 = (oof >= 0.5).astype(int)
    tn5, fp5, fn5, tp5 = confusion_matrix(y, pred_05, labels=[0, 1]).ravel()
    sens_05 = tp5 / (tp5 + fn5) if (tp5 + fn5) > 0 else 0.0
    spec_05 = tn5 / (tn5 + fp5) if (tn5 + fp5) > 0 else 0.0
    ppv_05  = tp5 / (tp5 + fp5) if (tp5 + fp5) > 0 else 0.0
    npv_05  = tn5 / (tn5 + fn5) if (tn5 + fn5) > 0 else 0.0

    print(f"\n  {MODEL_LABELS[key]}")
    print(f"  {'─'*58}")
    print(f"  Youden prah : {thr_opt:.3f}   J-index = {j_opt:.4f}   "
          f"F1 = {f1_opt:.3f}")
    print(f"  {'Metrika':<14} {'Hodnota':>8}  {'95% CI':^21}")
    print(f"  {'Senzitivita':<14} {sens_opt:>8.3f}  [{ci_sens[0]:.3f} – {ci_sens[1]:.3f}]")
    print(f"  {'Specificita':<14} {spec_opt:>8.3f}  [{ci_spec[0]:.3f} – {ci_spec[1]:.3f}]")
    print(f"  {'PPV':<14} {ppv_opt:>8.3f}  [{ci_ppv[0]:.3f} – {ci_ppv[1]:.3f}]")
    print(f"  {'NPV':<14} {npv_opt:>8.3f}  [{ci_npv[0]:.3f} – {ci_npv[1]:.3f}]")
    print(f"  Prah 0.5  →  Sens={sens_05:.3f}  Spec={spec_05:.3f}  "
          f"(Δ Sens={sens_opt-sens_05:+.3f}  Δ Spec={spec_opt-spec_05:+.3f})")

    thresh_rows.append({
        'Model':              MODEL_LABELS[key],
        'Youden_threshold':   round(thr_opt, 4),
        'Sensitivity_Youden': round(sens_opt, 4),
        'Sensitivity_CI_lo':  round(ci_sens[0], 4),
        'Sensitivity_CI_hi':  round(ci_sens[1], 4),
        'Specificity_Youden': round(spec_opt, 4),
        'Specificity_CI_lo':  round(ci_spec[0], 4),
        'Specificity_CI_hi':  round(ci_spec[1], 4),
        'PPV':                round(ppv_opt, 4),
        'PPV_CI_lo':          round(ci_ppv[0], 4),
        'PPV_CI_hi':          round(ci_ppv[1], 4),
        'NPV':                round(npv_opt, 4),
        'NPV_CI_lo':          round(ci_npv[0], 4),
        'NPV_CI_hi':          round(ci_npv[1], 4),
        'F1_Youden':          round(f1_opt, 4),
        'J_index':            round(j_opt, 4),
        'Sensitivity_05':     round(sens_05, 4),
        'Specificity_05':     round(spec_05, 4),
        'PPV_05':             round(ppv_05, 4),
        'NPV_05':             round(npv_05, 4),
    })

print()
print("  Interpretácia pre klinickú prax:")
print("  Vyšší prah → nižšia Senzitivita, vyššia Specificita (menej falošne pozitívnych)")
print("  Nižší prah → vyššia Senzitivita, nižšia Specificita (menej falošne negatívnych)")
print("  Pre synkopu je typicky dôležitejšia vysoká Senzitivita (nezmeškať pozitívny HUTT).")
print()
print("  Obmedzenie: Youden prah je vypočítaný z tých istých OOF dát na ktorých sa hodnotí.")
print("  Ide o odhad z interných dát — klinický prah vyžaduje nezávislé overenie.")

_val_sheets['prah_youden'] = pd.DataFrame(thresh_rows)
print("Pripravené: sheet 'prah_youden'")

# Aplikačný prah — fixný 0.5 pre všetky modely
# Youdenov prah (sekcia 4) slúži iba ako interný analytický ukazovateľ,
# nie ako aplikačný prah. Dôvod: Youden prah je optimalizovaný
# na tých istých OOF dátach, na ktorých sa hodnotí → optimistický odhad.
MODEL_APP_THR = {_r['Model']: 0.5 for _r in thresh_rows}


# =============================================================
# 5. SHAP HODNOTY — P3 Extra Trees
# =============================================================

print("\n" + "=" * 65)
print("5. SHAP HODNOTY — P3 Extra Trees")
print("=" * 65)

if not SHAP_OK:
    print("  Preskočené — shap nie je nainštalovaný (pip install shap).")
else:
    # Načítame finálny model uložený analyza.py — rovnaké CV-konsenzus features
    # ako pri trénovaní, žiaden rozdiel voči sekcii 4.
    _p3_pkg_path = 'model_p3_et.joblib'
    try:
        _p3_pkg = joblib.load('model_p3_et.joblib')
        _pipe_p3 = _p3_pkg['pipeline']

        # Vstupné stĺpce = to, čo pipeline očakáva na vstupe
        _input_cols = _p3_pkg.get('input_features', P3_POOL)

        # Prejdi cez imp + fs krok (bez clf) — presne to čo classifier dostal pri tréningu
        _imp_step = _pipe_p3.named_steps['imp']
        X_imp_shap = _imp_step.transform(df_feat[_input_cols])

        if 'fs' in _pipe_p3.named_steps:
            _fs_step = _pipe_p3.named_steps['fs']
            X_sel = _fs_step.transform(X_imp_shap)
            _fs_mask = _fs_step.get_support()
            sel_cols = [c for c, m in zip(_input_cols, _fs_mask) if m]
        else:
            X_sel = X_imp_shap
            sel_cols = list(_input_cols)

        et_final = _pipe_p3.named_steps['clf']
        print(f"  Načítané: model_p3_et.joblib")

    except FileNotFoundError:
        # fallback — trénuj náhradný model
        print("  UPOZORNENIE: model_p3_et.joblib neexistuje — tréning náhradného modelu...")
        X_p3 = df_feat[P3_POOL]
        imp = SimpleImputer(strategy='median')
        X_imp = imp.fit_transform(X_p3)
        fs = P3SelectorConsensus(n_p1=N_P1)
        fs.fit(X_imp, y)
        mask = fs.get_support()
        sel_cols = [c for c, m in zip(P3_POOL, mask) if m]
        X_sel = fs.transform(X_imp)
        et_final = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                        n_jobs=1, class_weight='balanced')
        et_final.fit(X_sel, y)

    print(f"  Vybraté features pre SHAP ({len(sel_cols)}): {', '.join(sel_cols)}")
    print("  Počítam SHAP hodnoty (TreeExplainer)...")

    explainer   = shap.TreeExplainer(et_final)
    shap_values = explainer.shap_values(X_sel)

    # Pre binárnu klasifikáciu: shap_values môže byť list [neg, pos]
    # alebo 3D pole (n_samples, n_features, n_classes) v novších verziách SHAP
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_pos = shap_values[:, :, 1]
    else:
        shap_pos = shap_values

    # Summary plot — beeswarm
    fig_shap, ax_shap = plt.subplots(figsize=(10, max(6, len(sel_cols) * 0.4)))
    shap.summary_plot(shap_pos, X_sel, feature_names=sel_cols,
                      show=False, plot_type='dot')
    plt.title('SHAP hodnoty — P3 Extra Trees ()', fontsize=12)
    plt.tight_layout()
    plt.savefig('validacia_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Uložené: validacia_shap_summary.png")

    # Bar plot — priemerné |SHAP|
    mean_abs = np.abs(shap_pos).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    fig_bar, ax_bar = plt.subplots(figsize=(9, max(5, len(sel_cols) * 0.35)))
    ax_bar.barh([sel_cols[i] for i in order[::-1]],
                [mean_abs[i] for i in order[::-1]],
                color='seagreen', alpha=0.8)
    ax_bar.set_xlabel('Priemerná |SHAP| hodnota')
    ax_bar.set_title('Priemerný vplyv features — P3 Extra Trees ()', fontsize=11)
    ax_bar.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('validacia_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Uložené: validacia_shap_bar.png")

    # CSV — priemerné |SHAP|
    shap_df = pd.DataFrame({
        'Feature':    [sel_cols[i] for i in order],
        'Mean_abs_SHAP': [round(mean_abs[i], 6) for i in order],
    })
    _val_sheets['shap'] = shap_df
    print("Pripravené: sheet 'shap'")

    print("\n  Top 5 features podľa SHAP (P3 ET):")
    for i in order[:5]:
        print(f"    {sel_cols[i]:<35} mean|SHAP| = {mean_abs[i]:.4f}")


# =============================================================
# 6. DECISION CURVE ANALYSIS (DCA)
# =============================================================

print("\n" + "=" * 65)
print("6. DECISION CURVE ANALYSIS (DCA)")
print("=" * 65)
print("  Net benefit(t) = TP/n − FP/n × t/(1−t)")
print("  Porovnanie s 'Liečiť všetkých' a 'Neliečiť nikoho'.")
print()

_dca_thresh = np.linspace(0.01, 0.95, 200)
_n          = len(y)
_prev       = y.mean()

fig_dca, ax_dca = plt.subplots(figsize=(10, 6))
ax_dca.axhline(0, color='black', lw=1.5, linestyle='--', label='Neliečiť nikoho (NB=0)')
_nb_all = _prev - (1 - _prev) * _dca_thresh / np.clip(1 - _dca_thresh, 1e-9, None)
ax_dca.plot(_dca_thresh, _nb_all, color='gray', lw=1.5, linestyle=':', label='Liečiť všetkých')

for key, oof in oof_probas.items():
    _nb_model = []
    for t in _dca_thresh:
        _pred_t = (oof >= t).astype(int)
        _tn_t, _fp_t, _fn_t, _tp_t = confusion_matrix(
            y, _pred_t, labels=[0, 1]).ravel()
        _nb_model.append(_tp_t / _n - _fp_t / _n * t / max(1 - t, 1e-9))
    ax_dca.plot(_dca_thresh, _nb_model, color=MODEL_COLORS[key],
                lw=2, label=MODEL_LABELS[key])
    print(f"  {MODEL_LABELS[key]:<25}  max NB = {max(_nb_model):.4f}  "
          f"(pri prahu {_dca_thresh[int(np.argmax(_nb_model))]:.2f})")

ax_dca.set_xlim([0, 0.95])
ax_dca.set_ylim([-0.05, min(0.6, _prev + 0.05)])
ax_dca.set_xlabel('Rozhodovací prah')
ax_dca.set_ylabel('Net benefit')
ax_dca.set_title('Decision Curve Analysis —  ConsensusFS', fontsize=12)
ax_dca.legend(fontsize=9)
ax_dca.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('validacia_dca.png', dpi=150, bbox_inches='tight')
plt.close()
print("Uložené: validacia_dca.png")


# =============================================================
# 7. PERMUTAČNÝ TEST  (štatistická signifikantnosť modelu)
# =============================================================

print("\n" + "=" * 65)
print("7. PERMUTAČNÝ TEST  (H₀: model nepredikuje lepšie ako náhoda)")
print("=" * 65)
print("  OOF predikcie sú fixné; permutujeme nálepky y (n=1000).")
print("  p-hodnota = podiel permutácií kde AUC_perm ≥ AUC_obs.")
print()

N_PERM   = 1000
perm_rng = np.random.default_rng(RANDOM_STATE + 99)

perm_rows = []
for key, oof in oof_probas.items():
    auc_obs   = roc_auc_score(y, oof)
    perm_aucs = []
    for _ in range(N_PERM):
        y_perm = perm_rng.permutation(y)
        try:
            perm_aucs.append(roc_auc_score(y_perm, oof))
        except ValueError:
            pass
    perm_aucs = np.array(perm_aucs)
    p_val  = float(np.mean(perm_aucs >= auc_obs))
    sig    = ("***" if p_val < 0.001
              else "**" if p_val < 0.01
              else "*"  if p_val < 0.05
              else "n.s.")
    _p_str = ("p < 0.001" if p_val < 0.001
              else f"p = {p_val:.3f}")
    print(f"  {MODEL_LABELS[key]:<25}  AUC_obs={auc_obs:.4f}  "
          f"AUC_perm={perm_aucs.mean():.4f}±{perm_aucs.std():.4f}  "
          f"{_p_str}  {sig}")
    perm_rows.append({
        'Model':       MODEL_LABELS[key],
        'AUC_obs':     round(auc_obs, 4),
        'AUC_perm_mean': round(float(perm_aucs.mean()), 4),
        'AUC_perm_std':  round(float(perm_aucs.std()), 4),
        'p_value':     round(p_val, 4),
        'significance': sig,
    })

print()
print("  Interpretácia: *** p<0.001  ** p<0.01  * p<0.05  n.s. p≥0.05")
_val_sheets['permutacny_test'] = pd.DataFrame(perm_rows)
print("Pripravené: sheet 'permutacny_test'")


# =============================================================
# 8. FAIRNESS ANALÝZA — subskupiny Pohlavie a Vek
# =============================================================

print("\n" + "=" * 65)
print("8. FAIRNESS ANALÝZA — Pohlavie a Vek")
print("=" * 65)
print("  Metriky sa počítajú na OOF predikciách pri fixnom aplikačnom prahu 0.5.")
print("  Rovnaký prah pre P1, P3 aj P4 — konzistentné porovnanie naprieč modelmi.")
print("  Pohlavie: F=0, M=1  |  Vek: medián rozdeľuje dve skupiny.")
print()

# subskupiny
_vek_med  = np.median(df_feat['Vek'].values) if 'Vek' in df_feat.columns else None
_subgroups = {}
if 'Pohlavie' in df_feat.columns:
    _subgroups['Pohlavie F'] = df_feat.index[df_feat['Pohlavie'] == 0].tolist()
    _subgroups['Pohlavie M'] = df_feat.index[df_feat['Pohlavie'] == 1].tolist()
if _vek_med is not None:
    _subgroups[f'Vek <{_vek_med:.0f}']  = df_feat.index[df_feat['Vek'] <  _vek_med].tolist()
    _subgroups[f'Vek ≥{_vek_med:.0f}']  = df_feat.index[df_feat['Vek'] >= _vek_med].tolist()

# reset index pre numpy indexovanie
_pos_map = {idx: pos for pos, idx in enumerate(df_feat.index)}


def _subgroup_metrics(y_sub, oof_sub, thr):
    pred = (oof_sub >= thr).astype(int)
    try:
        auc = roc_auc_score(y_sub, oof_sub)
    except ValueError:
        auc = float('nan')
    try:
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_sub, pred, labels=[0, 1]).ravel()
        sens = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else float('nan')
        spec = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else float('nan')
        ppv  = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else float('nan')
        npv  = tn_s / (tn_s + fn_s) if (tn_s + fn_s) > 0 else float('nan')
    except ValueError:
        sens = spec = ppv = npv = float('nan')
    return auc, sens, spec, ppv, npv


fairness_rows = []

for key, oof in oof_probas.items():
    thr      = MODEL_APP_THR[MODEL_LABELS[key]]
    thr_type = "fixný 0.5"

    print(f"  {MODEL_LABELS[key]}  (aplikačný prah={thr:.3f}, {thr_type})")
    print(f"  {'Skupina':<16} {'n':>4}  {'prev%':>6}  {'AUC':>6}  "
          f"{'Sens':>5}  {'Spec':>5}  {'PPV':>5}  {'NPV':>5}")
    print(f"  {'─'*62}")

    # celok pre referenziu
    auc_all, sens_all, spec_all, ppv_all, npv_all = _subgroup_metrics(y, oof, thr)
    print(f"  {'Celok':<16} {len(y):>4}  {y.mean()*100:>5.1f}%  "
          f"{auc_all:>6.3f}  {sens_all:>5.3f}  {spec_all:>5.3f}  "
          f"{ppv_all:>5.3f}  {npv_all:>5.3f}")

    for sg_name, sg_idx in _subgroups.items():
        pos_arr  = [_pos_map[i] for i in sg_idx if i in _pos_map]
        if len(pos_arr) < 10:
            print(f"  {sg_name:<16}  príliš malá skupina (n={len(pos_arr)}), preskočené.")
            continue
        y_sg   = y[pos_arr]
        oof_sg = oof[pos_arr]
        auc_sg, sens_sg, spec_sg, ppv_sg, npv_sg = _subgroup_metrics(y_sg, oof_sg, thr)

        def _fmt(v):
            return f"{v:5.3f}" if not np.isnan(v) else "  n/a"

        print(f"  {sg_name:<16} {len(pos_arr):>4}  {y_sg.mean()*100:>5.1f}%  "
              f"{_fmt(auc_sg):>6}  {_fmt(sens_sg):>5}  {_fmt(spec_sg):>5}  "
              f"{_fmt(ppv_sg):>5}  {_fmt(npv_sg):>5}")

        fairness_rows.append({
            'Model':    MODEL_LABELS[key],
            'Skupina':  sg_name,
            'n':        len(pos_arr),
            'prev_pct': round(y_sg.mean() * 100, 1),
            'AUC':      round(auc_sg, 4) if not np.isnan(auc_sg) else None,
            'Sensitivity': round(sens_sg, 4) if not np.isnan(sens_sg) else None,
            'Specificity': round(spec_sg, 4) if not np.isnan(spec_sg) else None,
            'PPV':      round(ppv_sg, 4) if not np.isnan(ppv_sg) else None,
            'NPV':      round(npv_sg, 4) if not np.isnan(npv_sg) else None,
        })
    print()

_val_sheets['fairness'] = pd.DataFrame(fairness_rows)
print("Pripravené: sheet 'fairness'")
print()
print("  Interpretácia: výrazný rozdiel AUC medzi skupinami (>0.10) signalizuje")
print("  nerovnomerný výkon modelu — relevantné pre klinické nasadenie.")


# =============================================================
# ZÁVEREČNÉ ZHRNUTIE
# =============================================================

print("\n" + "=" * 65)
print("ZÁVEREČNÉ ZHRNUTIE VALIDÁCIE")
print("=" * 65)
print(f"{'Model':<25} {'AUC':>6}  {'95% CI AUC':^17}  {'Brier':>6}  "
      f"{'Prah':>5}  {'Sens':>5}  {'Spec':>5}  {'PPV':>5}  {'NPV':>5}  {'p-perm':>7}")
print("─" * 105)

for key, oof in oof_probas.items():
    auc        = roc_auc_score(y, oof)
    mn, lo, hi = boot_results[key]
    brier      = brier_score_loss(y, oof)
    trow       = next(r for r in thresh_rows if r['Model'] == MODEL_LABELS[key])
    prow       = next(r for r in perm_rows   if r['Model'] == MODEL_LABELS[key])
    app_thr    = MODEL_APP_THR[MODEL_LABELS[key]]

    _sens = trow['Sensitivity_05']
    _spec = trow['Specificity_05']
    _ppv = trow['PPV_05']
    _npv = trow['NPV_05']
    print(f"{MODEL_LABELS[key]:<25} {auc:>6.4f}  "
          f"[{lo:.4f}–{hi:.4f}]  {brier:>6.4f}  "
          f"{app_thr:>5.3f}  "
          f"{_sens:>5.3f}  "
          f"{_spec:>5.3f}  "
          f"{_ppv:>5.3f}  "
          f"{_npv:>5.3f}  "
          f"{('<0.001' if prow['p_value'] < 0.001 else str(round(prow['p_value'],3))):>7} {prow['significance']}")

print()
print("  Pozn.: Prah — fixný 0.5 pre všetky modely (aplikačný).")
print("         Youdenov prah (sekcia 4) je analytický ukazovateľ, nie aplikačný prah.")
print("         Sens/Spec/PPV/NPV zodpovedajú aplikačnému prahu každého modelu.")
print()
print("Limitácie:")
print("  • Interná validácia — výsledky nie sú prenositeľné bez externej validácie.")
print("  • SHAP je vypočítaný na finálnom modeli trénovanom na celom datasete (interpretácia).")
print("  • Youden prah je štatisticky optimálny; klinický prah určuje lekár.")
print("  • Bootstrap CI zachytáva neistotu vzorky, nie variabilitu trénovania.")

# =============================================================
# ULOŽENIE DO EXCELU
# =============================================================

_EXCEL_VAL = 'validacia.xlsx'
with pd.ExcelWriter(_EXCEL_VAL, engine='openpyxl') as _writer:
    for _sheet, _df in _val_sheets.items():
        _df.to_excel(_writer, sheet_name=_sheet, index=False)
print(f"\nUložené: {_EXCEL_VAL}  (sheets: {', '.join(_val_sheets.keys())})")