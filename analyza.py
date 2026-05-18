"""
Predikcia výsledku HUTT testu — hutt_analysis.py
======================================================
Bakalárska práca: Predikcia výsledku HUTT testu

Verzia  — Feature selection: KONSENZUS Chi2 + RF + RFE (prahová logika, bez k)
  - ConsensusFeatureSelector VNÚTRI pipeline (žiaden leakage)
      • Chi2: features s p-value < 0.05 (po internom MinMaxScaling)
      • RF importance: features s importance > priemerná importance
      • RFE (LR): top sqrt(n_features) features, škálované RobustScalerom
        — fixný počet zvolený ako kompromis: dátovo nezávislý, no konzistentný
          s rozsahom (napr. sqrt(121)≈11 pre P2; sqrt(126)≈11 pre P3)
      • Konsenzus: feature ostáva ak ju nominovali ≥2 z 3 metód
        Fallback: ak by zostalo <3 features, akceptuje sa 1 hlas
  - Jednoúrovňové CV (5 foldov, žiaden inner CV — niet čo ladiť)
  - Tri pipeline: P1 anamnéza (bez FS), P2 dotazník, P3 kombinácia
  - Ensemble: P4 stacking (ET+KNN+LR → Ridge meta-learner, OOF anti-leakage)
  - Výstup: poradie modelov, vybrané atribúty, ROC, per-fold AUC
  - Prahová analýza, bootstrap CI, SHAP a DCA sú v separátnom skripte

Spustenie:
    python analyza.py
Požiadavky:
    pip install pandas numpy scikit-learn xgboost catboost matplotlib joblib
"""

import ast
from collections import Counter

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict)
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve,
                              f1_score)
from sklearn.base import BaseEstimator, TransformerMixin, clone
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RANDOM_STATE = 42
OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
INNER_CV  = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # len stacking (5b)
# : žiaden TUNE_K — prahová logika, počet features určujú dáta

# --- Prepínač agregátu srdcového ochorenia ---
# True  = aktuálne správanie: P1–P7 vypadnú, vznikne Ma_diag_srdcove_ochorenie
# False = P1–P7 ostanú v poole ako samostatné features, Ma_diag nevzniká
USE_MA_DIAG = False

_analyza_sheets: dict = {}   # collector — každý df sa uloží tu, na konci → analyza.xlsx

MODEL_COLORS = {
    'Logist. regresia':    '#888780',
    'Random Forest':       '#378ADD',
    'Extra Trees':         '#1D9E75',
    'XGBoost':             '#EF9F27',
    'CatBoost':            '#D85A30',
    'SVM':                 '#7F77DD',
    'KNN':                 '#E9A',

}


# =============================================================
# KONSENZUS FEATURE SELECTOR (vnútri Pipeline, žiaden leakage)
# =============================================================

class ConsensusFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection konsenzusom troch metód — prahová logika (bez k-limitu).

    Metódy:
      1. Chi2       — features s p-value < chi2_alpha (po MinMaxScaling)
      2. RF import  — features s importance > priemerná importance
      3. RFE (LR)   — sqrt(n_features) features (škálované RobustScalerom)

    Logika výberu:
      feature ostáva ak prešla ≥ min_votes (default 2) z 3 metód.
      Fallback: ak by zostalo < min_selected features, akceptuje sa 1 hlas.
    """

    def __init__(self, chi2_alpha=0.05, min_votes=2, min_selected=3):
        self.chi2_alpha   = chi2_alpha
        self.min_votes    = min_votes
        self.min_selected = min_selected

    def fit(self, X, y):
        import math
        n_features = X.shape[1]
        rfe_k = max(1, int(math.sqrt(n_features)))

        # --- Chi2: p-value < chi2_alpha ---
        X_mm = MinMaxScaler().fit_transform(X)
        _, chi2_pvals = chi2(X_mm, y)
        mask_chi2 = chi2_pvals < self.chi2_alpha

        # --- RF importance > priemer ---
        rf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=1, class_weight='balanced')
        rf.fit(X, y)
        mask_rf = rf.feature_importances_ > rf.feature_importances_.mean()

        # --- RFE s Logistickou regresiou (sqrt(n_features), škálované) ---
        X_rfe = RobustScaler().fit_transform(X)
        lr    = LogisticRegression(
            max_iter=300, random_state=RANDOM_STATE,
            class_weight='balanced', C=0.1)
        step  = max(1, n_features // 10)
        rfe   = RFE(estimator=lr, n_features_to_select=rfe_k, step=step)
        rfe.fit(X_rfe, y)
        mask_rfe = rfe.support_

        # --- Konsenzus ≥ min_votes ---
        votes   = mask_chi2.astype(int) + mask_rf.astype(int) + mask_rfe.astype(int)
        support = votes >= self.min_votes

        if support.sum() < self.min_selected:
            support = votes >= 1
        if support.sum() == 0:
            top_idx = np.argsort(votes)[::-1][:max(3, n_features // 10)]
            support = np.zeros(n_features, dtype=bool)
            support[top_idx] = True

        self.support_     = support
        self.votes_       = votes
        self.n_selected_  = int(support.sum())
        return self

    def transform(self, X):
        return X[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_)[0]
        return self.support_


# =============================================================
# 1. NAČÍTANIE DÁT
# =============================================================

print("=" * 65)
print("1. NAČÍTANIE DÁT  [hutt_analysis — Konsenzus FS (prahová logika)]")
print("=" * 65)

df = pd.read_csv('data_full1.csv')
print(f"Dataset: {df.shape[0]} pacientov, {df.shape[1]} stĺpcov")

a10_vals = df['A10'].replace(-1, np.nan).dropna()
print(f"A10 (cieľ): {a10_vals.value_counts().to_dict()}")
print(f"Pozitívnych (A10=1): {a10_vals.mean()*100:.1f}%")
print("""
Cieľová premenná: A10
  0 = HUTT negatívny  |  1 = HUTT pozitívny
Feature selection: ConsensusFeatureSelector (Chi2+RF+RFE, ≥2/3) vnútri CV

POZOR — interpretácia výsledkov:
  Model predikuje pravdepodobnosť pozitívneho výsledku HUTT testu
  z predtestových údajov — nie diagnózu synkopy, nie klinický stav pacienta.
  Výsledky predstavujú INTERNÚ validáciu na jednom datasete.
  Externá validácia na nezávislej kohorte je nevyhnutná pred klinickým nasadením.
""")


# =============================================================
# 2. PREDSPRACOVANIE  (identické s 05.py)
# =============================================================

print("=" * 65)
print("2. PREDSPRACOVANIE")
print("=" * 65)

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
    # P1–P7 sa vyhodia, nahrádza ich agregát
    ALL_REMOVE = (admin_cols + leakage_cols + post_test_cols +
                  flag_cols + poor_quality_cols + text_cols +
                  doctor_remove_cols + p_srdc_cols)
else:
    # P1–P7 ostávajú v poole ako samostatné binárne features
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
    n_pos = (df_work['Ma_diag_srdcove_ochorenie'] == 1).sum()
    n_neg = (df_work['Ma_diag_srdcove_ochorenie'] == 0).sum()
    print(f"Ma_diag_srdcove_ochorenie (agregát P1–P7): pos={n_pos} neg={n_neg}")
else:
    print(f"P1–P7 ostávajú ako samostatné features (USE_MA_DIAG=False)")
    for c in p_srdc:
        n1 = (df_work[c] == 1).sum()
        print(f"  {c}: pos={n1}")

df_work = df_work.drop(columns=ALL_REMOVE, errors='ignore')

if 'A2' in df_work.columns:
    bp = df_work['A2'].str.split('/', expand=True)
    df_work['A2_sys'] = pd.to_numeric(bp[0], errors='coerce')
    df_work['A2_dia'] = pd.to_numeric(
        bp[1] if bp.shape[1] > 1 else pd.Series([np.nan] * len(df_work)),
        errors='coerce')
    df_work.drop(columns=['A2'], inplace=True)
    print("A2 → A2_sys, A2_dia")

df_work['Pohlavie'] = df_work['Pohlavie'].map({'F': 0, 'M': 1})
for col in ['B1', 'J3']:
    if col in df_work.columns:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

_n_before_drop = len(df_work)
df_work = df_work.dropna(subset=['A10'])
_n_dropped = _n_before_drop - len(df_work)
if _n_dropped > 0:
    print(f"Vyradení pacienti: {_n_dropped} (chýbajúca hodnota A10 — "
          f"buď pôvodne -1 alebo skutočne chýbajúce)")
df_work['A10'] = df_work['A10'].astype(int)

y       = df_work['A10'].values
df_feat = df_work.drop(columns=['A10']).copy()

print(f"Po predspracovaní: {len(y)} pacientov, {df_feat.shape[1]} features")
print(f"A10=1: {y.sum()} ({y.mean()*100:.1f}%)  A10=0: {(1-y).sum()}")



# =============================================================
# 3. FEATURE POOLS
# =============================================================

print("\n" + "=" * 65)
print("3. FEATURE POOLS")
print("=" * 65)

P1_COLS = [c for c in ['A2_sys', 'A2_dia', 'A3', 'Vek', 'Pohlavie']
           if c in df_feat.columns]
a_cols   = [c for c in df_feat.columns if c.startswith('A')]
dem_cols = ['Vek', 'Pohlavie']
P2_POOL  = [c for c in df_feat.columns if c not in a_cols + dem_cols]
P3_POOL  = P1_COLS + [c for c in P2_POOL if c not in P1_COLS]  # P1 prvé!

print(f"P1 (anamnéza):   {len(P1_COLS)} features — žiadna FS")
print(f"P2 (dotazník):   {len(P2_POOL)} features v poole → ConsensusFS (prahová logika)")
print(f"P3 (kombinácia): 5 passthrough + ConsensusFS (prahová logika) z {len(P2_POOL)} dotazníkových")
if USE_MA_DIAG:
    print(f"Ma_diag_srdcove_ochorenie v P2 poole: "
          f"{'ÁNO' if 'Ma_diag_srdcove_ochorenie' in P2_POOL else 'NIE'}")
    print(f"Ma_diag_srdcove_ochorenie v P3 poole: "
          f"{'ÁNO' if 'Ma_diag_srdcove_ochorenie' in P3_POOL else 'NIE'}")
else:
    in_p2 = [c for c in p_srdc_cols if c in P2_POOL]
    print(f"P1–P7 v P2 poole (USE_MA_DIAG=False): {in_p2 if in_p2 else 'žiadne — skontroluj'}")

POOLS = {
    'P1_anamneza':   P1_COLS,
    'P2_dotaznik':   P2_POOL,
    'P3_kombinacia': P3_POOL,
}
# : žiadne K_CANDIDATES — prahová logika (Chi2 p<0.05, RF>mean, RFE sqrt(n))
N_P1 = len(P1_COLS)


# =============================================================
# 4. PIPELINE FACTORY (s ConsensusFeatureSelector)
# =============================================================

print("\n" + "=" * 65)
print("4. PIPELINE FACTORY  [ConsensusFeatureSelector]")
print("=" * 65)

pos_weight = float((y == 0).sum()) / float((y == 1).sum())


class P3SelectorConsensus(BaseEstimator, TransformerMixin):
    """
    P3 feature selector: prvých n_p1 stĺpcov = P1 passthrough (vždy),
    zvyšok = ConsensusFeatureSelector (prahová logika) z dotazníkového poolu.
    Výstup: [P1_cols | vybrané_P2_cols]
    """
    def __init__(self, n_p1, min_votes=2):
        self.n_p1      = n_p1
        self.min_votes = min_votes

    def fit(self, X, y):
        X_p2     = X[:, self.n_p1:]
        self.fs_ = ConsensusFeatureSelector(min_votes=self.min_votes)
        self.fs_.fit(X_p2, y)
        return self

    def transform(self, X):
        return np.hstack([X[:, :self.n_p1],
                          self.fs_.transform(X[:, self.n_p1:])])

    def get_support(self):
        return np.concatenate([np.ones(self.n_p1, dtype=bool),
                               self.fs_.get_support()])


def make_pipe(clf, apply_fs=False, scale=False, n_p1=0):
    """
    sklearn Pipeline:
      1. SimpleImputer (median)
      2. FS krok (ak apply_fs=True):
           n_p1=0  → ConsensusFeatureSelector (prahová logika)  pre P2
           n_p1>0  → P3SelectorConsensus(n_p1)                 pre P3
      3. RobustScaler — voliteľné (SVM, LR)
      4. Klasifikátor
    """
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


def classifiers():
    return {
        'Logist. regresia': (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                               class_weight='balanced', C=0.1), True),
        'Random Forest': (
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                   n_jobs=1, class_weight='balanced'), False),
        'Extra Trees': (
            ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                 n_jobs=1, class_weight='balanced'), False),
        'XGBoost': (
            XGBClassifier(n_estimators=150, random_state=RANDOM_STATE,
                          eval_metric='logloss', verbosity=0,
                          scale_pos_weight=pos_weight), False),
        'CatBoost': (
            CatBoostClassifier(iterations=100, random_seed=RANDOM_STATE,
                               verbose=0, auto_class_weights='Balanced'), False),
        'SVM': (
            SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE,
                class_weight='balanced'), True),
        'KNN': (
            KNeighborsClassifier(n_neighbors=7, weights='distance',
                                 metric='euclidean'), True),
    }

print("Klasifikátory:", list(classifiers().keys()))
print("FS metóda: ConsensusFeatureSelector — prahová logika (Chi2 p<0.05, RF>mean, RFE sqrt(n))")
print("  Chi2 poznámka: aplikovaný po internom MinMaxScalovaní na [0,1].")
print("  Väčšina P2 premenných je binárna (0/1) → Chi2 je priamočiara voľba.")
print("  Numerické premenné (C1,C2,C4) sú po transformácii tiež v [0,1] — skríningová metóda.")


# =============================================================
# 5. NESTED CV 5×5
# =============================================================

print("\n" + "=" * 65)
print("5. JEDNOÚROVŇOVÉ CV (5 foldov)  [konsenzus FS — prahová logika]")
print("=" * 65)
print("AUC = nezaujaté (model nevidel testovací fold počas trénovania)")
print("FS:  Chi2 p<0.05  |  RF > mean importance  |  RFE top sqrt(n) features")
print()


def cv_pipeline(clf, scale, X_df, y_arr, apply_fs=False, n_p1=0):
    """
    Jednoúrovňové 5-fold CV s ConsensusFS vnútri každého foldu (žiaden inner CV).
    apply_fs=True: P2/P3 → ConsensusFeatureSelector (prahová logika, bez k).
    apply_fs=False: P1 → žiadna FS (5 features, všetky vstupujú do modelu).
    Vracia: fold_aucs, oof_proba, [] (žiadne k), fold_features.
    """
    oof_proba     = np.zeros(len(y_arr))
    fold_aucs     = []
    fold_features = []
    all_cols      = list(X_df.columns)

    for tr_idx, te_idx in OUTER_CV.split(X_df, y_arr):
        X_tr = X_df.iloc[tr_idx]
        X_te = X_df.iloc[te_idx]
        y_tr = y_arr[tr_idx]
        y_te = y_arr[te_idx]

        pipe_best = make_pipe(clone(clf), apply_fs=apply_fs, scale=scale, n_p1=n_p1)
        pipe_best.fit(X_tr, y_tr)
        proba             = pipe_best.predict_proba(X_te)[:, 1]
        oof_proba[te_idx] = proba
        fold_aucs.append(roc_auc_score(y_te, proba))

        step_names = [n for n, _ in pipe_best.steps]
        if 'fs' in step_names:
            mask = pipe_best['fs'].get_support()
            sel  = [c for c, m in zip(all_cols, mask) if m]
        else:
            sel  = all_cols[:]
        fold_features.append(sel)

    return np.array(fold_aucs), oof_proba, [], fold_features


all_results      = []
all_fold_aucs    = {}
all_oof_proba    = {}
all_fold_features = {}   # per-fold vybrané atribúty

for pname, pool_cols in POOLS.items():
    X_pipe    = df_feat[pool_cols]
    apply_fs  = (pname != 'P1_anamneza')   # P1 bez FS
    clfs      = classifiers()

    print(f"{'─'*65}")
    print(f"{pname} ({len(pool_cols)} features v poole)")
    print(f"{'Model':<22} {'F1':>6} {'F2':>6} {'F3':>6} {'F4':>6} {'F5':>6}"
          f"  {'Mean':>6} {'Std':>5}  Záver")
    print("─" * 65)

    n_p1_pipe = N_P1 if pname == 'P3_kombinacia' else 0

    for mname, (clf_obj, need_scale) in clfs.items():
        fold_aucs, oof_proba, chosen_ks, fold_feats = cv_pipeline(
            clf_obj, need_scale, X_pipe, y,
            apply_fs=apply_fs, n_p1=n_p1_pipe)

        mean_auc = fold_aucs.mean()
        std_auc  = fold_aucs.std(ddof=1)
        verdict  = ("konzistentne" if std_auc < 0.03
                    else "akceptovatelne" if std_auc < 0.05
                    else "variabilne")

        _oof_pred = (oof_proba >= 0.5).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(y, _oof_pred).ravel()
        _sens = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
        _spec = _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0
        _f1   = f1_score(y, _oof_pred, zero_division=0)

        row = {
            'Pipeline':    pname,
            'Model':       mname,
            'AUC_CV_mean':  round(mean_auc, 4),
            'AUC_std':     round(std_auc,  4),
            'Sensitivity': round(_sens, 4),
            'Specificity': round(_spec, 4),
            'F1':          round(_f1,   4),
            'Fold1':  round(fold_aucs[0], 4),
            'Fold2':  round(fold_aucs[1], 4),
            'Fold3':  round(fold_aucs[2], 4),
            'Fold4':  round(fold_aucs[3], 4),
            'Fold5':  round(fold_aucs[4], 4),
            'K_chosen': '—',
        }
        all_results.append(row)
        all_fold_aucs[(pname, mname)]     = fold_aucs
        all_oof_proba[(pname, mname)]     = oof_proba
        all_fold_features[(pname, mname)] = fold_feats

        fstr = " ".join(f"{v:.3f}" for v in fold_aucs)
        print(f"{mname:<22} {fstr}  {mean_auc:.3f}  {std_auc:.3f}  {verdict}")

    print()

results_df = pd.DataFrame(all_results)


# Ensemble konštanty
RUN_ENSEMBLE      = True   # nastavte False pre rýchly beh bez P4
_PNAME_STACK      = 'P4_stacking'
_MNAME_STACK      = 'Stacking (ET+KNN+LR→Ridge)'
_STACK_BASE_NAMES = ['Extra Trees', 'KNN', 'Logist. regresia']

# =============================================================
# 5b. STACKING — P4 (P3: ET + KNN + LR → Ridge meta-learner)
# =============================================================
if RUN_ENSEMBLE:
    print("\n" + "=" * 65)
    print("5b. STACKING  P4 = Ridge(ET_proba, KNN_proba, LR_proba)")
    print("=" * 65)
    print("Feature space: P3 (kombinácia, ConsensusFS)")
    print("Base modely:   Extra Trees  +  KNN  +  Logist. regresia")
    print("Meta-learner:  Ridge Logistická regresia (L2, OOF vstupy)")
    print()

    X_p3_s = df_feat[P3_POOL]

    _n_base         = len(_STACK_BASE_NAMES)
    oof_proba_stack = np.zeros(len(y))
    fold_aucs_stack = []

    for fold_i, (tr_idx, te_idx) in enumerate(OUTER_CV.split(X_p3_s, y), 1):
        X_tr = X_p3_s.iloc[tr_idx];  X_te = X_p3_s.iloc[te_idx]
        y_tr = y[tr_idx];            y_te = y[te_idx]

        # OOF meta-features pre tréningovú časť (vnútorná CV)
        meta_tr = np.zeros((len(tr_idx), _n_base))
        for in_tr_rel, in_val_rel in INNER_CV.split(X_tr, y_tr):
            for bi, bname in enumerate(_STACK_BASE_NAMES):
                _clf_b, _ns_b = classifiers()[bname]
                _pipe_b = make_pipe(_clf_b, apply_fs=True, scale=_ns_b, n_p1=N_P1)
                _pipe_b.fit(X_tr.iloc[in_tr_rel], y_tr[in_tr_rel])
                meta_tr[in_val_rel, bi] = \
                    _pipe_b.predict_proba(X_tr.iloc[in_val_rel])[:, 1]

        # Ridge meta-learner trénovaný na OOF meta-features
        _meta_clf = LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs',
            max_iter=500, random_state=RANDOM_STATE)
        _meta_clf.fit(meta_tr, y_tr)

        # Base modely trénované na celom tréningovom folde → predikcia na test
        meta_te = np.zeros((len(te_idx), _n_base))
        for bi, bname in enumerate(_STACK_BASE_NAMES):
            _clf_b, _ns_b = classifiers()[bname]
            _pipe_b = make_pipe(_clf_b, apply_fs=True, scale=_ns_b, n_p1=N_P1)
            _pipe_b.fit(X_tr, y_tr)
            meta_te[:, bi] = _pipe_b.predict_proba(X_te)[:, 1]

        fold_proba = _meta_clf.predict_proba(meta_te)[:, 1]
        oof_proba_stack[te_idx] = fold_proba
        fold_aucs_stack.append(roc_auc_score(y_te, fold_proba))

        coef_str = '  '.join(
            f'{bname[:3]}={_meta_clf.coef_[0, bi]:.3f}'
            for bi, bname in enumerate(_STACK_BASE_NAMES))
        print(f"  Fold {fold_i}: AUC={fold_aucs_stack[-1]:.4f}  meta coef: {coef_str}")

    _mean_s = np.mean(fold_aucs_stack)
    _std_s  = np.std(fold_aucs_stack, ddof=1)
    _pred_s = (oof_proba_stack >= 0.5).astype(int)
    _tn, _fp, _fn, _tp = confusion_matrix(y, _pred_s).ravel()
    _sens_s = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
    _spec_s = _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0
    _f1_s   = f1_score(y, _pred_s, zero_division=0)

    _oof_auc_s = roc_auc_score(y, oof_proba_stack)
    print(f"\n{_PNAME_STACK}  ·  {_MNAME_STACK}")
    print(f"  CV AUC (priemer foldov): {_mean_s:.4f} ± {_std_s:.4f}")
    print(f"  OOF AUC (celé dáta):     {_oof_auc_s:.4f}  "
          f"[OOF AUC počítaný naraz na všetkých dátach — môže sa mierne líšiť od priemeru foldov]")
    print(f"  OOF@0.5: Sens={_sens_s:.4f}  Spec={_spec_s:.4f}  F1={_f1_s:.4f}")

    _stack_row = {
        'Pipeline':    _PNAME_STACK,
        'Model':       _MNAME_STACK,
        'AUC_CV_mean':  round(_mean_s, 4),
        'AUC_std':     round(_std_s,  4),
        'Sensitivity': round(_sens_s, 4),
        'Specificity': round(_spec_s, 4),
        'F1':          round(_f1_s,   4),
        'Fold1': round(fold_aucs_stack[0], 4),
        'Fold2': round(fold_aucs_stack[1], 4),
        'Fold3': round(fold_aucs_stack[2], 4),
        'Fold4': round(fold_aucs_stack[3], 4),
        'Fold5': round(fold_aucs_stack[4], 4),
        'K_chosen': '—',
    }
    all_results.append(_stack_row)
    all_oof_proba[(_PNAME_STACK, _MNAME_STACK)] = oof_proba_stack
    all_fold_aucs[(_PNAME_STACK, _MNAME_STACK)] = np.array(fold_aucs_stack)

else:
    print("\n[5b PRESKOČENÉ — RUN_ENSEMBLE=False  (P4 stacking)]")

results_df = pd.DataFrame(all_results)


# =============================================================
# 6. PORADIE MODELOV
# =============================================================

print("=" * 65)
print("6. PORADIE MODELOV (podľa AUC_CV_mean)")
print("=" * 65)
print(f"{'Rank':<4} {'Pipeline':<16} {'Model':<22} {'AUC':>6} {'±Std':>6} "
      f"{'Sens':>6} {'Spec':>6} {'F1':>6}")
print("─" * 78)

ranked = results_df.sort_values('AUC_CV_mean', ascending=False).reset_index(drop=True)
for i, r in ranked.iterrows():
    marker = " ★" if i == 0 else (" ◆" if i == 1 else "")
    print(f"{i+1:<4} {r['Pipeline']:<16} {r['Model']:<22} "
          f"{r['AUC_CV_mean']:>6.3f} ±{r['AUC_std']:>5.3f} "
          f"{r['Sensitivity']:>6.3f} {r['Specificity']:>6.3f} {r['F1']:>6.3f}{marker}")

_ACTIVE_PIPES = (['P1_anamneza', 'P2_dotaznik', 'P3_kombinacia'] +
                 ([_PNAME_STACK] if RUN_ENSEMBLE else []))

print("\nNAJLEPSÍ PER PIPELINE:")
best_per_pipe = {}
for pipe in _ACTIVE_PIPES:
    pb = results_df[results_df['Pipeline'] == pipe].sort_values(
        'AUC_CV_mean', ascending=False).iloc[0]
    best_per_pipe[pipe] = pb
    print(f"  {pipe:<20}: {pb['Model']:<22} "
          f"AUC={pb['AUC_CV_mean']:.3f} ±{pb['AUC_std']:.3f}  "
          f"Sens={pb['Sensitivity']:.3f}  Spec={pb['Specificity']:.3f}  F1={pb['F1']:.3f}")
print("  [Sens/Spec/F1 sú orientačné hodnoty pri prahu 0.5 — klinicky optimálny prah")
print("   je predmetom prahovej analýzy v separátnom validačnom skripte.]")

# --- Súhrn vybraných atribútov ---
def _feat_summary(fold_feats, min_folds=3):
    """Konsenzus features vybrané v ≥ min_folds foldoch."""
    if not fold_feats:
        return 0, []
    freq = Counter(f for fold in fold_feats for f in fold)
    sel  = sorted([f for f, cnt in freq.items() if cnt >= min_folds],
                  key=lambda f: -freq[f])
    return len(sel), sel

_PIPE_FEATS = {}
for _pf in ['P1_anamneza', 'P2_dotaznik', 'P3_kombinacia']:
    _pb2 = best_per_pipe[_pf]
    _ff  = all_fold_features.get((_pf, _pb2['Model']), [])
    _PIPE_FEATS[_pf] = _feat_summary(_ff)
if RUN_ENSEMBLE:
    # P4 stacking je na P3 — features = zjednotenie konsenzusových features
    # všetkých 3 base modelov (ET, KNN, LR) na P3
    _p4_all = []
    for _bname in _STACK_BASE_NAMES:
        _ff_b = all_fold_features.get(('P3_kombinacia', _bname), [])
        _, _sel_b = _feat_summary(_ff_b)
        _p4_all += [f for f in _sel_b if f not in _p4_all]
    _PIPE_FEATS[_PNAME_STACK] = (len(_p4_all), _p4_all)

print("\nVYBRANÉ ATRIBÚTY per pipeline (konsenzus ≥3/5 foldov):")
print("─" * 65)
for pipe in _ACTIVE_PIPES:
    n_f, feats = _PIPE_FEATS[pipe]
    pb = best_per_pipe[pipe]
    print(f"\n  {pipe}  ·  {pb['Model']}  →  {n_f} atribútov:")
    for i in range(0, len(feats), 5):
        print(f"    {', '.join(feats[i:i+5])}")

# ---------------------------------------------------------
# 6b. VYBRANÉ ATRIBÚTY — najlepší model každej pipeline
# ---------------------------------------------------------
print("\n" + "─" * 65)
print("6b. VYBRANÉ ATRIBÚTY (per fold, najlepší model každej pipeline)")
print("─" * 65)

feat_selection_rows = []
for pipe in ['P1_anamneza', 'P2_dotaznik', 'P3_kombinacia']:
    pb      = best_per_pipe[pipe]
    mname   = pb['Model']
    key     = (pipe, mname)
    folds   = all_fold_features[key]

    print(f"\n{pipe}  ·  {mname}")

    from collections import Counter as _Ctr
    freq = _Ctr(f for fold in folds for f in fold)
    n_folds = len(folds)

    # Per-fold výpis
    for fi, fold_feats in enumerate(folds, 1):
        print(f"  Fold {fi} (k={len(fold_feats)}): {', '.join(fold_feats)}")

    # Konsenzus: features vybrané v ≥3/5 foldoch
    consensus = sorted([f for f, cnt in freq.items() if cnt >= 3],
                       key=lambda f: -freq[f])
    print(f"\n  Konsenzus (≥3/5 foldov, {len(consensus)} features):")
    for f in consensus:
        ma = ' ← Ma_diag' if f == 'Ma_diag_srdcove_ochorenie' else ''
        print(f"    {f:<35} vybraná v {freq[f]}/{n_folds} foldoch{ma}")

    # Sledovanie klinicky zaujímavých premenných
    if pipe != 'P1_anamneza':
        if USE_MA_DIAG:
            ma_cnt = freq.get('Ma_diag_srdcove_ochorenie', 0)
            print(f"\n  Ma_diag_srdcove_ochorenie: vybraná v {ma_cnt}/{n_folds} foldoch"
                  f"  {'→ PREŠLA FS ✓' if ma_cnt > 0 else '→ NEPREŠLA FS ✗'}")
        else:
            srdc_selected = {c: freq.get(c, 0) for c in p_srdc_cols if c in freq}
            if srdc_selected:
                print(f"\n  P1–P7 (srdcové ochorenie, jednotlivé):")
                for c, cnt in sorted(srdc_selected.items(), key=lambda x: -x[1]):
                    print(f"    {c}: vybraná v {cnt}/{n_folds} foldoch"
                          f"  {'✓' if cnt > 0 else '✗'}")

    # Uloženie do CSV
    for f, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        feat_selection_rows.append({
            'Pipeline': pipe, 'Model': mname,
            'Feature': f, 'Folds_selected': cnt, 'Total_folds': n_folds,
            'Consensus': cnt >= 3,
        })

_analyza_sheets['features'] = pd.DataFrame(feat_selection_rows)
print("\nPripravené: sheet 'features'")



# =============================================================
# 7. GRAFY
# =============================================================

print("\n" + "=" * 65)
print("7. GRAFY")
print("=" * 65)


# ROC krivky — najlepší model každej pipeline
_roc_pipes  = ['P1_anamneza', 'P2_dotaznik', 'P3_kombinacia']
_roc_colors = ['royalblue', 'darkorange', 'seagreen']
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle('ROC krivky — najlepší model každej pipeline [ConsensusFS (prahová logika)]', fontsize=12)
for ax, pipe, col in zip(axes1, _roc_pipes, _roc_colors):
    pb      = best_per_pipe[pipe]
    mname   = pb['Model']
    oof     = all_oof_proba[(pipe, mname)]
    fpr_arr, tpr_arr, _ = roc_curve(y, oof)
    auc_val  = roc_auc_score(y, oof)          # OOF AUC (≠ priemer foldov)
    cv_auc   = pb['AUC_CV_mean']
    ax.plot(fpr_arr, tpr_arr, color=col, lw=2,
            label=f'{mname}\nOOF AUC={auc_val:.3f}  CV AUC={cv_auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax.set_title(pipe)
    ax.set_xlabel('1 − Specificita')
    ax.set_ylabel('Sensitivita')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hutt_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Uložené: hutt_roc.png")


# Per-fold AUC variabilita
model_names_plot = list(classifiers().keys())
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig2.suptitle('CV AUC per fold [ConsensusFS (prahová logika)]', fontsize=13)
for ax, pname in zip(axes2, ['P1_anamneza', 'P2_dotaznik', 'P3_kombinacia']):
    for i, mname in enumerate(model_names_plot):
        folds = all_fold_aucs.get((pname, mname))
        if folds is None:
            continue
        mean  = folds.mean()
        std   = folds.std()
        color = MODEL_COLORS.get(mname, 'gray')
        ax.vlines(i, folds.min(), folds.max(), color=color, lw=2, alpha=0.5)
        for fv, jit in zip(folds, np.linspace(-0.08, 0.08, len(folds))):
            ax.scatter(i + jit, fv, color=color, s=30, zorder=5, alpha=0.85)
        ax.scatter(i, mean, color=color, s=120, zorder=6,
                   edgecolors='white', linewidths=1.5)
        ax.fill_between([i - 0.25, i + 0.25], mean - std, mean + std,
                        color=color, alpha=0.12)
    ax.set_xticks(range(len(model_names_plot)))
    ax.set_xticklabels(model_names_plot, rotation=35, ha='right', fontsize=8)
    ax.set_title(pname, fontsize=11)
    ax.set_ylim([0.35, 1.00])
    ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    if ax == axes2[0]:
        ax.set_ylabel('AUC')
patches = [mpatches.Patch(color=c, label=m)
           for m, c in MODEL_COLORS.items() if m in model_names_plot]
fig2.legend(handles=patches, loc='lower center', ncol=6,
            fontsize=8, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('hutt_cv_folds.png', dpi=150, bbox_inches='tight')
plt.close()
print("Uložené: hutt_cv_folds.png")


# =============================================================
# 8. ULOŽENIE VÝSLEDKOV
# =============================================================

print("\n" + "=" * 65)
print("8. ULOŽENIE")
print("=" * 65)

results_df['FS'] = 'ConsensusFS'
_analyza_sheets['vysledky'] = results_df.sort_values('AUC_CV_mean', ascending=False)
print("Pripravené: sheet 'vysledky'")

fold_rows = []
for (pipe, mname), folds in all_fold_aucs.items():
    fold_rows.append({
        'Pipeline': pipe, 'Model': mname, 'FS': 'ConsensusFS',
        'Mean_AUC_CV': round(folds.mean(), 4),
        'AUC_std':         round(folds.std(ddof=1),  4),
        'Fold1': round(folds[0], 4), 'Fold2': round(folds[1], 4),
        'Fold3': round(folds[2], 4), 'Fold4': round(folds[3], 4),
        'Fold5': round(folds[4], 4),
    })
_analyza_sheets['per_fold'] = pd.DataFrame(fold_rows)
print("Pripravené: sheet 'per_fold'")
print("sheet 'features'  (pripravený v sekcii 6b)")


# =============================================================
# ZÁVEREČNÉ ZHRNUTIE — TOP 3 MODELY
# =============================================================

print("\n" + "=" * 65)
print("ZÁVEREČNÉ ZHRNUTIE — TOP 3 MODELY")
print("=" * 65)
print(f"Dataset: {len(y)} pacientov  |  A10=1: {y.sum()}  A10=0: {(1-y).sum()}")
print(f"Feature selection: ConsensusFS (Chi2+RF+RFE, ≥2/3)\n")

print(f"{'Rank':<4} {'Pipeline':<18} {'Model':<22} {'AUC':>6} {'±Std':>6} "
      f"{'Sens':>6} {'Spec':>6} {'F1':>6}")
print("─" * 82)
for i, r in ranked.head(3).iterrows():
    marker = " ★" if i == 0 else (" ◆" if i == 1 else " ●")
    print(f"{i+1:<4} {r['Pipeline']:<18} {r['Model']:<22} "
          f"{r['AUC_CV_mean']:>6.3f} ±{r['AUC_std']:>5.3f} "
          f"{r['Sensitivity']:>6.3f} {r['Specificity']:>6.3f} {r['F1']:>6.3f}{marker}")

print("\nVYBRANÉ ATRIBÚTY — TOP 3 MODELY (konsenzus ≥3/5 foldov):")
print("─" * 65)
for i, r in ranked.head(3).iterrows():
    pipe   = r['Pipeline']
    n_f, feats = _PIPE_FEATS.get(pipe, (0, []))
    print(f"\n  #{i+1}  {pipe}  ·  {r['Model']}  →  {n_f} atribútov:")
    for j in range(0, len(feats), 6):
        print(f"    {', '.join(feats[j:j+6])}")


# =============================================================
# ULOŽENIE MODELOV PRE APLIKÁCIU  (P1 ET + P3 ET)
# =============================================================

print("\n" + "=" * 65)
print("ULOŽENIE MODELOV PRE APLIKÁCIU  (P1 ET + P3 ET)")
print("=" * 65)


def _thr_metrics(y_true, oof, thr):
    pred = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return sens, spec, ppv, npv


def _youden_thr(y_true, oof):
    fpr, tpr, thresholds = roc_curve(y_true, oof)
    idx = np.argmax(tpr + (1 - fpr) - 1)
    return float(thresholds[idx])


_APP_MODELS = [
    # (pname,            mname,          pool,     apply_fs, n_p1,  fixed_thr, fname)
    ('P1_anamneza',   'Extra Trees', P1_COLS,  False, 0,    0.500, 'model_p1_et.joblib'),
    ('P3_kombinacia', 'Extra Trees', P3_POOL,  True,  N_P1, None,  'model_p3_et.joblib'),
]

_saved_pkgs = {}   # uložíme pkg-y pre terminálový dotazník

for pname, mname, pool, apply_fs, n_p1_s, fixed_thr, fname in _APP_MODELS:
    key = (pname, mname)
    if key not in all_oof_proba:
        print(f"  SKIP {fname} — OOF chýbajú pre {key}")
        continue

    oof_m    = all_oof_proba[key]
    res_row  = results_df[
        (results_df['Pipeline'] == pname) & (results_df['Model'] == mname)
    ].iloc[0]

    youden   = _youden_thr(y, oof_m)
    use_thr  = fixed_thr if fixed_thr is not None else youden

    sens, spec, ppv, npv       = _thr_metrics(y, oof_m, use_thr)
    sens05, spec05, _, _       = _thr_metrics(y, oof_m, 0.5)

    # CV-konsenzus features (≥3/5 foldov) — zmrazená sada pre finálny model
    # Zabezpečuje súlad medzi reportovanými atribútmi a aplikačným modelom.
    if pname in _PIPE_FEATS and _PIPE_FEATS[pname][1]:
        cv_feats = _PIPE_FEATS[pname][1]
    else:
        cv_feats = list(pool)   # fallback: P1 (bez FS) alebo prázdny konsenzus

    p2_sel = [f for f in cv_feats if f not in P1_COLS]

    # Tréning finálneho modelu len na CV-konsenzus features (bez nového FS kroku)
    clf_f  = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                   n_jobs=1, class_weight='balanced')
    pipe_f = make_pipe(clf_f, apply_fs=False, scale=False, n_p1=0)
    pipe_f.fit(df_feat[cv_feats], y)

    pkg = {
        'pipeline':             pipe_f,
        'features':             cv_feats,
        'selected_features':    cv_feats,
        'cv_consensus_features': cv_feats,
        'p2_selected_features': p2_sel,
        'threshold':            round(use_thr, 4),
        'threshold_youden':     round(youden, 4),
        'model_name':           mname,
        'pipeline_name':        pname,
        'n_train':              len(y),
        'n_pos':                int(y.sum()),
        'AUC_CV_mean':          round(float(res_row['AUC_CV_mean']), 4),
        'AUC_std':              round(float(res_row['AUC_std']), 4),
        'sensitivity_at_thr':   round(sens, 4),
        'specificity_at_thr':   round(spec, 4),
        'ppv_at_thr':           round(ppv, 4),
        'npv_at_thr':           round(npv, 4),
        'sensitivity_05':       round(sens05, 4),
        'specificity_05':       round(spec05, 4),
        'train_proba_pos':      oof_m[y == 1].tolist(),
        'train_proba_neg':      oof_m[y == 0].tolist(),
    }
    joblib.dump(pkg, fname)
    _saved_pkgs[pname] = pkg

    # Porovnávacia tabuľka: CV konsenzus vs finálny model (teraz identické)
    cv_set    = set(cv_feats)
    final_set = set(cv_feats)
    print(f"  Uložené: {fname}")
    print(f"    CV-konsenzus features (≥3/5 foldov): {len(cv_feats)}")
    for f in cv_feats:
        print(f"      {f}")
    print(f"    Finálny model = CV-konsenzus features ✓  (žiadny rozdiel)")
    print(f"    P2 selected: {len(p2_sel)}")
    print(f"    Prah: {use_thr:.3f}  (Youden: {youden:.3f})")
    print(f"    Sens={sens:.3f}  Spec={spec:.3f}  PPV={ppv:.3f}  NPV={npv:.3f}")
    print(f"    AUC_CV_mean={res_row['AUC_CV_mean']:.4f} ± {res_row['AUC_std']:.4f}")


# =============================================================
# ULOŽENIE DO EXCELU
# =============================================================

_EXCEL_ANALYZA = 'analyza.xlsx'
with pd.ExcelWriter(_EXCEL_ANALYZA, engine='openpyxl') as _writer:
    for _sheet, _df in _analyza_sheets.items():
        _df.to_excel(_writer, sheet_name=_sheet, index=False)
print(f"\nUložené: {_EXCEL_ANALYZA}  (sheets: {', '.join(_analyza_sheets.keys())})")
