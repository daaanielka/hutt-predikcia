# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
================================================================================
  Predikcia výsledku HUTT testu – 10f (metodologické vylepšenia oproti 10d)
================================================================================

  Vylepšenia oproti 10d:
    1. NESTED CV – vonkajší 5-fold CV pre nestranný odhad výkonu,
       vnútorný 5-fold CV pre výber K. Žiadne kontaminácia testovacieho setu
       pri výbere modelu.
    2. SEPARÁTNY VÝBER PRAHU – prah sa volí na VALIDAČNOM folde vnútorného CV,
       nie na testovacom sete. Testovací set sa používa VÝLUČNE pre finálne
       vyhodnotenie.
    3. DECISION CURVE ANALYSIS (DCA) – čistý prínos vs. pravdepodobnostný prah.
       Klinicky relevantnejšie ako samotné AUC.

  Cieľová premenná: A10 = výsledok HUTT testu (1=pozitívny, 0=negatívny)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, joblib
warnings.filterwarnings('ignore')

from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, VotingClassifier,
                               ExtraTreesClassifier, HistGradientBoostingClassifier,
                               BaggingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sklearn.base as skbase

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_full1.csv')
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
SEED      = 42
np.random.seed(SEED)

# ==============================================================================
# NASTAVENIA
# ==============================================================================
MIN_SENS_SKRINING = 0.95
N_ANA             = 5   # počet anamnestických atribútov – vždy zahrnuté

APP_ANA_MODEL = "ExtraTrees"
APP_KOM_MODEL = "RF"

# discrete_features='auto': sklearn automaticky rozlíši diskrétne (binárne checkbox) vs. spojité
# (C1=vek pri výskyte, C2=počet odpadnutí, C4=vek v období ťažkostí) podľa hodnôt.
# Pôvodné discrete_features=True bolo nesprávne pre C1/C2/C4, ktoré sú numerické.
# Anamnestické spojité premenné (Vek, TK, Pulz) prechádzajú passthrough – MI sa na ne neaplikuje.
MI_SCORE = partial(mutual_info_classif, random_state=SEED, discrete_features='auto')

# ==============================================================================
# SEKCIA 1: NAČÍTANIE A PREDSPRACOVANIE
# ==============================================================================
print("\n" + "="*70)
print("  10f_hutt_predikcia.py")
print("  Predikcia vysledku HUTT testu (A10: 1=pozitivny, 0=negativny)")
print("  NOVINKA: Nested CV + separatny vyber prahu + Decision Curve Analysis")
print("="*70)

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

df = pd.read_csv(DATA_PATH)
print(f"\nNacitanych: {len(df)} pacientov, {len(df.columns)} stlpcov")

bp = df["A2"].map(parse_bp)
df["TK_sys"]       = [x[0] for x in bp]
df["TK_dia"]       = [x[1] for x in bp]
df["Pulz"]         = pd.to_numeric(df["A3"], errors="coerce").replace(-1, np.nan)
df["Pohlavie_enc"] = (df["Pohlavie"] == "M").astype(float)

P_SRDC = [c for c in ['P1','P2','P3','P4','P5','P6','P7','P8'] if c in df.columns]
df["Ma_diag_srdcove_ochorenie"] = df[P_SRDC].replace(-1, np.nan).max(axis=1)
print(f"Ma_diag_srdcove_ochorenie vytvoreny z: {P_SRDC}")

df["A10_clean"] = pd.to_numeric(df["A10"], errors="coerce")
df_valid = df[df["A10_clean"].notna()].copy()
y_all = df_valid["A10_clean"].astype(int).values

n_pos = y_all.sum(); n_neg = (y_all==0).sum()
print(f"\nCielova premenna: A10 = vysledok HUTT testu")
print(f"  A10=1 (pozitivny): {n_pos} ({n_pos/len(y_all)*100:.1f}%)")
print(f"  A10=0 (negativny): {n_neg} ({n_neg/len(y_all)*100:.1f}%)")
print(f"  Celkovo: {len(y_all)}")

n_missing_a10 = df["A10_clean"].isna().sum()
if n_missing_a10 > 0:
    print(f"  POZOR: {n_missing_a10} pacientov bez A10 – vyluceni")

synkopa_col = pd.to_numeric(df_valid["Synkopa"], errors="coerce")
if synkopa_col.notna().sum() > 0:
    agreement = (y_all == synkopa_col.fillna(-1).astype(int)).mean()
    print(f"\n  Zhoda A10 vs Synkopa = {agreement*100:.1f}% → Synkopa vylucena (leakage)")

META = {"Pohlavie","Pohlavie_enc","Vek","Synkopa","Typ Synkopy",
        "A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A10_clean",
        "TK_sys","TK_dia","Pulz","Datum","Datum narodenia","S",
        "Cislo dotaznika","Dátum","Číslo dotazníka"}

# Oprava imputácie: checkbox vs. numerické premenné
df_valid = df_valid.copy()
_checkbox_cols, _numeric_cols = [], []
for col in df_valid.columns:
    if col in META:
        continue
    col_num = pd.to_numeric(df_valid[col], errors="coerce")
    non_missing = col_num[col_num != -1].dropna()
    unique_nonmiss = set(non_missing.unique()) if len(non_missing) > 0 else set()
    if unique_nonmiss.issubset({0.0, 1.0}):
        df_valid[col] = col_num.replace(-1, 0).fillna(0)
        _checkbox_cols.append(col)
    else:
        df_valid[col] = col_num.replace(-1, np.nan)
        _numeric_cols.append(col)

print(f"\nImputacia: {len(_checkbox_cols)} checkbox stlpcov (-1→0),"
      f"  {len(_numeric_cols)} numerickych stlpcov (-1→NaN)")

# ==============================================================================
# SEKCIA 2: SKUPINY ATRIBÚTOV + SPLIT 80/20
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 2: SKUPINY ATRIBUTOV + SPLIT 80/20 (seed=42)")
print("="*70)

VYLUCENE = {'B2','C3','J1','N7','P32',
            'Q1','Q4','Q12','Q13','Q16','Q17','Q18',
            'A10','A10_clean','Synkopa','Typ Synkopy'}

ANAMNEZA = ['Pohlavie_enc','Vek','TK_sys','TK_dia','Pulz']

DOTAZNIK_KAND = [c for c in df_valid.columns
                 if c not in META
                 and c not in VYLUCENE
                 and c not in {"Ma_diag_srdcove_ochorenie"}
                 and not c.startswith("A")]
DOTAZNIK_KAND = DOTAZNIK_KAND + ["Ma_diag_srdcove_ochorenie"]

KOMBINACIA_KAND = ANAMNEZA + DOTAZNIK_KAND

print(f"Anamneza (vzdy zahrnutych): {len(ANAMNEZA)} atributov → indexy 0-{N_ANA-1}")
print(f"Dotaznik kandidati: {len(DOTAZNIK_KAND)} atributov → indexy {N_ANA}+")
print(f"Kombinacia celkom: {len(KOMBINACIA_KAND)} atributov")
print(f"\nVylucene: Synkopa, Typ Synkopy (data leakage), A10 (ciel)")

y = y_all
_bin_mask = (y == 0) | (y == 1)
_bin_idx  = np.where(_bin_mask)[0]
_y_bin    = y[_bin_idx]
n_nonbin  = len(y) - len(_bin_idx)
if n_nonbin > 0:
    print(f"  Vyluceni pacienti s A10 ∉ {{0,1}}: {n_nonbin}")

# Rovnaký split ako 10d (seed=42) pre porovnateľné výsledky
_rel_tr, _rel_te = train_test_split(
    np.arange(len(_bin_idx)), test_size=0.2, random_state=SEED, stratify=_y_bin
)
tr_idx = _bin_idx[_rel_tr]
te_idx = _bin_idx[_rel_te]
y_tr = y[tr_idx]; y_te = y[te_idx]
print(f"\nTrain: {len(tr_idx)}  |  Test: {len(te_idx)}")
print(f"Test: A10=1: {y_te.sum()}, A10=0: {(y_te==0).sum()}")

# VarianceThreshold – odstrán konštantné atribúty (len tréningové dáta)
from sklearn.feature_selection import VarianceThreshold as _VT
_X_vt = df_valid[DOTAZNIK_KAND].values[tr_idx].astype(float)
_imp_vt = SimpleImputer(strategy='median')
_X_vt_imp = _imp_vt.fit_transform(_X_vt)
_vt = _VT(threshold=0)
_vt.fit(_X_vt_imp)
_vt_removed = [f for f, keep in zip(DOTAZNIK_KAND, _vt.get_support()) if not keep]
if _vt_removed:
    print(f"\n[VarianceThreshold] Konstantne atributy vylucene z DOTAZNIK_KAND:")
    for f in _vt_removed:
        print(f"  {f}")
    DOTAZNIK_KAND   = [f for f in DOTAZNIK_KAND if f not in _vt_removed]
    KOMBINACIA_KAND = ANAMNEZA + DOTAZNIK_KAND
    print(f"[VarianceThreshold] Aktualizovanych kandidatov: {len(DOTAZNIK_KAND)}")
else:
    print("\n[VarianceThreshold] Ziadne konstantne atributy nenajdene.")
del _X_vt, _imp_vt, _X_vt_imp, _vt, _vt_removed

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ==============================================================================
# SEKCIA 3: PIPELINE FUNKCIE
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 3: SKLEARN PIPELINE FUNKCIE")
print("  Anamneza:   SimpleImputer → CalibratedCV")
print("  Dotaznik:   SimpleImputer → SelectKBest(MI,k) → CalibratedCV")
print("  Kombinacia: SimpleImputer → ColumnTransformer(")
print("                passthrough[0..4], SelectKBest(MI,k)[5+]")
print("              ) → CalibratedCV")
print("="*70)

def make_pipeline_anamneza(base_clf, needs_scale=False):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if needs_scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', CalibratedClassifierCV(base_clf, method='isotonic', cv=3)))
    return Pipeline(steps)

def make_pipeline_dotaznik(base_clf, k, needs_scale=False):
    steps = [
        ('imputer',  SimpleImputer(strategy='median')),
        ('selector', SelectKBest(MI_SCORE, k=k)),
    ]
    if needs_scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', CalibratedClassifierCV(base_clf, method='isotonic', cv=3)))
    return Pipeline(steps)

def make_pipeline_kombinacia(base_clf, k_dot, n_dot, needs_scale=False):
    """
    Kombinacia: anamnestické (idx 0..N_ANA-1) VZDY passthrough,
    dotazníkové (idx N_ANA..N_ANA+n_dot-1) → SelectKBest(MI, k=k_dot).
    """
    ana_idx = list(range(N_ANA))
    dot_idx = list(range(N_ANA, N_ANA + n_dot))
    k_safe  = min(k_dot, len(dot_idx))
    ct = ColumnTransformer([
        ('ana', 'passthrough', ana_idx),
        ('dot', SelectKBest(MI_SCORE, k=k_safe), dot_idx),
    ])
    steps = [
        ('imputer',  SimpleImputer(strategy='median')),
        ('selector', ct),
    ]
    if needs_scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', CalibratedClassifierCV(base_clf, method='isotonic', cv=3)))
    return Pipeline(steps)

# Definície modelov (identické ako 10d)
_rf   = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=SEED)
_et   = ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=SEED)
_lr   = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, class_weight='balanced')
_svm  = LinearSVC(C=1.0, max_iter=2000, random_state=SEED, class_weight='balanced')
_gb   = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08,
                                    subsample=0.8, random_state=SEED)
_hgb  = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.08,
                                        random_state=SEED)
_ada  = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=SEED)
_xgb  = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08,
                       subsample=0.8, colsample_bytree=0.8,
                       use_label_encoder=False, eval_metric='logloss',
                       random_state=SEED, verbosity=0)
_lgbm = LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, verbose=-1)
_cat  = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.08,
                            eval_metric='AUC', random_seed=SEED,
                            verbose=0, allow_writing_files=False)
_svc_rbf = SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED)
_mlp  = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED,
                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
_bnb  = BernoulliNB(alpha=1.0)
_knn3 = KNeighborsClassifier(n_neighbors=3)
_knn7 = KNeighborsClassifier(n_neighbors=7)
_bag  = BaggingClassifier(estimator=LogisticRegression(max_iter=500, random_state=SEED),
                           n_estimators=50, random_state=SEED)
_v_rf_xgb = VotingClassifier(estimators=[
    ('rf',  RandomForestClassifier(n_estimators=200, max_depth=12, random_state=SEED)),
    ('xgb', XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08,
                           use_label_encoder=False, eval_metric='logloss',
                           random_state=SEED, verbosity=0)),
], voting='soft')
_v_rf_lgbm = VotingClassifier(estimators=[
    ('rf',   RandomForestClassifier(n_estimators=200, max_depth=12, random_state=SEED)),
    ('lgbm', LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.08,
                             random_state=SEED, verbose=-1)),
], voting='soft')
_v_triple = VotingClassifier(estimators=[
    ('rf',   RandomForestClassifier(n_estimators=150, max_depth=12, random_state=SEED)),
    ('xgb',  XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08,
                            use_label_encoder=False, eval_metric='logloss',
                            random_state=SEED, verbosity=0)),
    ('lgbm', LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.08,
                             random_state=SEED, verbose=-1)),
], voting='soft')
_stack = StackingClassifier(
    estimators=[
        ('rf',   RandomForestClassifier(n_estimators=150, max_depth=10, random_state=SEED)),
        ('xgb',  XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08,
                                use_label_encoder=False, eval_metric='logloss',
                                random_state=SEED, verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.08,
                                 random_state=SEED, verbose=-1)),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=SEED),
    cv=3, passthrough=False,
)

MODEL_DEFS = [
    ("RF",             _rf,       False),
    ("ExtraTrees",     _et,       False),
    ("GradBoost",      _gb,       False),
    ("HistGradBoost",  _hgb,      False),
    ("AdaBoost",       _ada,      False),
    ("XGBoost",        _xgb,      False),
    ("LightGBM",       _lgbm,     False),
    ("CatBoost",       _cat,      False),
    ("LR",             _lr,       True),
    ("LinearSVM",      _svm,      True),
    ("SVC-RBF",        _svc_rbf,  True),
    ("KNN-3",          _knn3,     True),
    ("KNN-7",          _knn7,     True),
    ("BernoulliNB",    _bnb,      False),
    ("MLP",            _mlp,      True),
    ("Bagging-LR",     _bag,      False),
    ("Voting RF+XGB",  _v_rf_xgb,  False),
    ("Voting RF+LGBM", _v_rf_lgbm, False),
    ("Voting 3x",      _v_triple,  False),
    ("Stacking",       _stack,     False),
]
BASELINE = DummyClassifier(strategy='most_frequent', random_state=SEED)

# Nested CV pre všetky modely – nestranný odhad výkonu bez model selection bias
NESTED_CV_MODELS = [m for m, _, _ in MODEL_DEFS]

# ==============================================================================
# SEKCIA 3b: K-GRID A POMOCNÉ FUNKCIE
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 3b: K-GRID + NESTED CV POMOCNE FUNKCIE")
print("  Vonkajsi 5-fold CV: nestranný odhad výkonu")
print("  Vnutorny 5-fold CV: vyber K per fold (bez kontaminacie testovaciaho setu)")
print("="*70)

n_dot = len(DOTAZNIK_KAND)
K_GRID = [3, 5, 8, 10, 12, 15, 17, 20, 25, 30, 40]
K_GRID = sorted(set(k for k in K_GRID if k <= n_dot))
print(f"\n  K-grid: {K_GRID}")
print(f"  Kandidati: Dotaznik={n_dot}, Kombinacia={n_dot} (+{N_ANA} anamnestickych vzdy)")

def najdi_k_opt(model, needs_scale, X_tr_k, y_tr_k, k_grid, gtype, n_dot_local, cv):
    """
    Pre daný model nájde optimálne K cez CV (greedy K-grid prehľad).
    Vracia: (best_k, best_auc_cv)
    """
    best_k, best_auc = k_grid[0], 0.0
    for k in k_grid:
        clf_k = skbase.clone(model)
        if gtype == "dotaznik":
            pipe_k = make_pipeline_dotaznik(clf_k, k=k, needs_scale=needs_scale)
        else:
            pipe_k = make_pipeline_kombinacia(clf_k, k_dot=k, n_dot=n_dot_local,
                                              needs_scale=needs_scale)
        sc = cross_val_score(pipe_k, X_tr_k, y_tr_k, cv=cv,
                             scoring='roc_auc', n_jobs=1).mean()
        if sc > best_auc:
            best_auc, best_k = sc, k
    return best_k, best_auc

def vyber_prah_na_val(proba_val, y_val, min_sens=MIN_SENS_SKRINING):
    """
    Vyberie prah na VALIDAČNOM sete (nie testovacom).
    Kritérium: najvyššia špecificita pri Sens >= min_sens.
    Fallback: Youdenov J.
    """
    thrs = np.arange(0.05, 0.96, 0.01)
    best_sk = {'spec': -1, 'thr': None}
    for t in thrs:
        pred = (proba_val >= t).astype(int)
        TP = int(((pred==1)&(y_val==1)).sum())
        FP = int(((pred==1)&(y_val==0)).sum())
        FN = int(((pred==0)&(y_val==1)).sum())
        TN = int(((pred==0)&(y_val==0)).sum())
        sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
        spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
        if sens >= min_sens and spec > best_sk['spec']:
            best_sk = {'spec': spec, 'thr': t}
    if best_sk['thr'] is not None:
        return best_sk['thr']
    # Fallback: Youdenov J
    best_j = {'j': -99, 'thr': 0.5}
    for t in thrs:
        pred = (proba_val >= t).astype(int)
        TP = int(((pred==1)&(y_val==1)).sum())
        FP = int(((pred==1)&(y_val==0)).sum())
        FN = int(((pred==0)&(y_val==1)).sum())
        TN = int(((pred==0)&(y_val==0)).sum())
        sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
        spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
        j = sens + spec - 1
        if j > best_j['j']:
            best_j = {'j': j, 'thr': t}
    return best_j['thr']

# ==============================================================================
# SEKCIA 4: EVALUAČNÉ FUNKCIE
# ==============================================================================
def calc_metrics(y_true, y_prob, thr=0.50):
    pred = (y_prob >= thr).astype(int)
    TP = int(((pred==1)&(y_true==1)).sum()); FP = int(((pred==1)&(y_true==0)).sum())
    FN = int(((pred==0)&(y_true==1)).sum()); TN = int(((pred==0)&(y_true==0)).sum())
    sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
    spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    f1   = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0.0
    acc  = (TP+TN)/len(y_true)
    return {"AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0.5,
            "Sens":sens,"Spec":spec,"F1":f1,"Acc":acc,
            "TP":TP,"FP":FP,"FN":FN,"TN":TN}

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    np.random.seed(seed); n = len(y_true); scores = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2: continue
        scores.append(metric_fn(yt, ys))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

# ==============================================================================
# SEKCIA 3c: NESTED CV (vonkajší 5-fold, vnútorný 5-fold)
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 3c: NESTED CV (vonkajsi 5-fold × vnutorny 5-fold)")
print("  Modely: RF, ExtraTrees, LightGBM, XGBoost, CatBoost")
print("  Skupina: Kombinacia (anamneza + SelectKBest dotaznik)")
print("  POZOR: Toto moze trvat 30-60 minut!")
print("="*70)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+1)

X_kom_full_tr = df_valid[KOMBINACIA_KAND].values[tr_idx].astype(float)

# POZOR: Nested CV prebieha len na trénovacej sade (80 % datasetu, n=len(tr_idx)).
# Každý vonkajší fold má validačnú vzorku ~n/5. Toto obmedzenie treba uviesť v texte práce.
print(f"  POZOR: Nested CV prebieha na n={len(tr_idx)} vzorkách (80 % datasetu).")
print(f"  Každý vonkajší fold má n_val ≈ {len(tr_idx)//5} vzoriek – interpretovať s opatrnosťou.")

nested_results = {}   # {model_name: {'fold_aucs':[], 'fold_sens':[], 'fold_spec':[], 'fold_thrs':[]}}

for mname, base_clf, needs_scale in MODEL_DEFS:
    if mname not in NESTED_CV_MODELS:
        continue

    print(f"\n  [Nested CV] {mname} ...")
    fold_aucs  = []
    fold_sens  = []
    fold_spec  = []
    fold_thrs  = []
    fold_k     = []

    for fold_i, (outer_tr, outer_val) in enumerate(outer_cv.split(X_kom_full_tr, y_tr)):
        X_out_tr = X_kom_full_tr[outer_tr]; y_out_tr = y_tr[outer_tr]
        X_out_val= X_kom_full_tr[outer_val]; y_out_val= y_tr[outer_val]

        # Vnútorný CV: výber K
        best_k_inner, best_auc_inner = K_GRID[0], 0.0
        for k in K_GRID:
            clf_k = skbase.clone(base_clf)
            pipe_k = make_pipeline_kombinacia(clf_k, k_dot=k, n_dot=n_dot,
                                              needs_scale=needs_scale)
            sc = cross_val_score(pipe_k, X_out_tr, y_out_tr,
                                 cv=inner_cv, scoring='roc_auc', n_jobs=1).mean()
            if sc > best_auc_inner:
                best_auc_inner, best_k_inner = sc, k

        # Tréning finálneho modelu na outer_train s best K
        clf_final = skbase.clone(base_clf)
        pipe_final = make_pipeline_kombinacia(clf_final, k_dot=best_k_inner,
                                              n_dot=n_dot, needs_scale=needs_scale)
        pipe_final.fit(X_out_tr, y_out_tr)

        # Predikcia na outer_val
        proba_val = pipe_final.predict_proba(X_out_val)[:, 1]

        # Výber prahu na outer_val (nie na test sete!)
        thr_val = vyber_prah_na_val(proba_val, y_out_val)

        # Metriky na outer_val
        if len(np.unique(y_out_val)) > 1:
            auc_val = roc_auc_score(y_out_val, proba_val)
        else:
            auc_val = 0.5
        m_val = calc_metrics(y_out_val, proba_val, thr=thr_val)

        fold_aucs.append(auc_val)
        fold_sens.append(m_val['Sens'])
        fold_spec.append(m_val['Spec'])
        fold_thrs.append(thr_val)
        fold_k.append(best_k_inner)

        print(f"    Fold {fold_i+1}: K={best_k_inner}  AUC={auc_val*100:.1f}%  "
              f"Prah={thr_val:.2f}  Sens={m_val['Sens']*100:.1f}%  "
              f"Spec={m_val['Spec']*100:.1f}%")

    nested_results[mname] = {
        'fold_aucs':  fold_aucs,
        'fold_sens':  fold_sens,
        'fold_spec':  fold_spec,
        'fold_thrs':  fold_thrs,
        'fold_k':     fold_k,
        'mean_auc':   np.mean(fold_aucs),
        'std_auc':    np.std(fold_aucs),
        'mean_sens':  np.mean(fold_sens),
        'std_sens':   np.std(fold_sens),
        'mean_spec':  np.mean(fold_spec),
        'std_spec':   np.std(fold_spec),
        'mean_thr':   np.mean(fold_thrs),
    }
    print(f"    → Nested CV súhrn: AUC={np.mean(fold_aucs)*100:.1f}% ± "
          f"{np.std(fold_aucs)*100:.1f}%  "
          f"Sens={np.mean(fold_sens)*100:.1f}%  "
          f"Spec={np.mean(fold_spec)*100:.1f}%")

# Uloženie nested CV výsledkov
nested_rows = []
for mname, res in nested_results.items():
    for fi in range(len(res['fold_aucs'])):
        nested_rows.append({
            'Model': mname,
            'Fold': fi+1,
            'AUC_%': round(res['fold_aucs'][fi]*100, 1),
            'Sens_%': round(res['fold_sens'][fi]*100, 1),
            'Spec_%': round(res['fold_spec'][fi]*100, 1),
            'Threshold': round(res['fold_thrs'][fi], 3),
            'K_opt': res['fold_k'][fi],
        })
    nested_rows.append({
        'Model': mname,
        'Fold': 'MEAN',
        'AUC_%': round(res['mean_auc']*100, 1),
        'Sens_%': round(res['mean_sens']*100, 1),
        'Spec_%': round(res['mean_spec']*100, 1),
        'Threshold': round(res['mean_thr'], 3),
        'K_opt': '',
    })
    nested_rows.append({
        'Model': mname,
        'Fold': 'STD',
        'AUC_%': round(res['std_auc']*100, 1),
        'Sens_%': round(res['std_sens']*100, 1),
        'Spec_%': '',
        'Threshold': '',
        'K_opt': '',
    })

pd.DataFrame(nested_rows).to_csv(
    os.path.join(OUT_DIR, 'vysledky_10f_nested_cv.csv'), index=False)
print(f"\n  Ulozene: vysledky_10f_nested_cv.csv")

# ==============================================================================
# SEKCIA 3d: WILCOXON SIGNED-RANK TEST (porovnanie top modelov v nested CV)
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 3d: WILCOXON SIGNED-RANK TEST (top modely, nested CV foldy)")
print("  POZOR: n=5 foldov → test má nízku štatistickú silu (min p=0.0625)")
print("="*70)

# Zoraď modely podľa mean_auc z nested CV
_sorted_nested = sorted(nested_results.items(),
                        key=lambda x: x[1]['mean_auc'], reverse=True)
print(f"\n  Poradie modelov podľa nested CV AUC (Kombinacia skupina):")
for i, (mn, res) in enumerate(_sorted_nested, 1):
    print(f"  {i:2d}. {mn:<20} AUC={res['mean_auc']*100:.1f}% ± {res['std_auc']*100:.1f}%")

# Wilcoxon test: top model vs. ostatné top 5
if len(_sorted_nested) >= 2:
    print(f"\n  Wilcoxon test: najlepší model vs. ostatní (5 foldov, two-sided)")
    _best_name, _best_res = _sorted_nested[0]
    _best_aucs = np.array(_best_res['fold_aucs'])
    wilcoxon_rows = []
    for mn, res in _sorted_nested[1:]:
        _other_aucs = np.array(res['fold_aucs'])
        try:
            stat, pval = wilcoxon(_best_aucs, _other_aucs, alternative='two-sided')
        except Exception:
            stat, pval = float('nan'), float('nan')
        diff_mean = (_best_res['mean_auc'] - res['mean_auc']) * 100
        print(f"  {_best_name} vs {mn:<20} Δ={diff_mean:+.1f}pp  p={pval:.4f}"
              f"{'  *' if pval < 0.05 else '  ns'}")
        wilcoxon_rows.append({'Model_A': _best_name, 'Model_B': mn,
                              'Delta_pp': round(diff_mean, 2),
                              'Wilcoxon_stat': round(stat, 3) if not np.isnan(stat) else '',
                              'p_value': round(pval, 4) if not np.isnan(pval) else '',
                              'Significant': pval < 0.05 if not np.isnan(pval) else False})
    pd.DataFrame(wilcoxon_rows).to_csv(
        os.path.join(OUT_DIR, 'vysledky_10f_wilcoxon.csv'), index=False)
    print(f"  Ulozene: vysledky_10f_wilcoxon.csv")
    print(f"  Poznamka: Pri n=5 foldoch je minimalne dosiahnutelne p = 0.0625 (two-sided).")

# Boxplot Nested CV AUC
try:
    if nested_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#F8F7F5"); ax.set_facecolor("#F8F7F5")
        names_nc = list(nested_results.keys())
        data_nc  = [nested_results[m]['fold_aucs'] for m in names_nc]
        bp_obj = ax.boxplot(
            [[v*100 for v in d] for d in data_nc],
            labels=names_nc, patch_artist=True, notch=False
        )
        colors_bp = ['#2E86AB','#27ae60','#e67e22','#e74c3c','#8e44ad']
        for patch, color in zip(bp_obj['boxes'], colors_bp[:len(names_nc)]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='AUC=70%')
        ax.set_ylabel("AUC (%) – vonkajší fold", fontsize=11)
        ax.set_title("Nested CV – distribúcia AUC cez 5 vonkajších foldov\n"
                     "(nestranný odhad – prah vybraný na validačnom folde, nie na test sete)",
                     fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'graf_10f_nested_cv.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Ulozeny: graf_10f_nested_cv.png")
except Exception as e:
    print(f"  Graf nested CV: {e}")

# ==============================================================================
# SEKCIA 5: FINÁLNY MODEL (tréning na 80% train sete)
# ==============================================================================
# Po nested CV máme nestranný odhad výkonu.
# Teraz trénujeme finálny model na celom train sete (80%).
#   - K_opt nájdeme cez CV na train sete (ako v 10d)
#   - Prah nájdeme na validačnom folde train CV (NIE na test sete)
#   - Finálne vyhodnotenie na held-out test sete (20%)
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 5: PIPELINE CV + TEST – 3 SKUPINY x 20 MODELOV + BASELINE")
print("  NOVINKA: Prah vybrany na validacnom folde CV (nie na test sete)")
print("="*70)

GROUPS = {
    "Anamneza":   {"feats": ANAMNEZA,        "type": "anamneza"},
    "Dotaznik":   {"feats": DOTAZNIK_KAND,   "type": "dotaznik"},
    "Kombinacia": {"feats": KOMBINACIA_KAND, "type": "kombinacia"},
}

all_results = {}
rows_csv    = []

for gname, gcfg in GROUPS.items():
    feats = gcfg["feats"]
    gtype = gcfg["type"]
    X_tr  = df_valid[feats].iloc[tr_idx].values.astype(float)
    X_te  = df_valid[feats].iloc[te_idx].values.astype(float)

    if gtype == "kombinacia":
        print(f"\n  -- {gname}: 5 anamnestickych (vzdy) + SelectKBest (K per-model CV) --")
    elif gtype == "dotaznik":
        print(f"\n  -- {gname}: {len(feats)} kandidatov → SelectKBest (K per-model CV) --")
    else:
        print(f"\n  -- {gname}: {len(feats)} atributov (vsetky, bez selekcie) --")

    all_results[gname] = {}

    def _make_pipe(clf, ns, k=None, _gtype=gtype):
        if _gtype == "anamneza":
            return make_pipeline_anamneza(clf, ns)
        elif _gtype == "dotaznik":
            return make_pipeline_dotaznik(clf, k, ns)
        else:
            return make_pipeline_kombinacia(clf, k, len(DOTAZNIK_KAND), ns)

    # Baseline
    pipe_base = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', BASELINE)])
    cv_base = cross_val_score(pipe_base, X_tr, y_tr, cv=cv5, scoring='roc_auc', n_jobs=1)
    pipe_base.fit(X_tr, y_tr)
    prob_base = pipe_base.predict_proba(X_te)[:,1]
    m_base = calc_metrics(y_te, prob_base)
    print(f"    {'Baseline':<16}  AUC_test={m_base['AUC']*100:.1f}%  "
          f"CV={cv_base.mean()*100:.1f}+/-{cv_base.std()*100:.1f}%  [REF]")
    all_results[gname]['Baseline'] = {
        'AUC_test':m_base['AUC'],'AUC_CV':cv_base.mean(),'AUC_CV_std':cv_base.std(),
        'Sens':m_base['Sens'],'Spec':m_base['Spec'],'F1':m_base['F1'],
        'TP':m_base['TP'],'FP':m_base['FP'],'FN':m_base['FN'],'TN':m_base['TN'],
        'proba_te':prob_base
    }

    for mname, base_clf, needs_scale in MODEL_DEFS:
        # Per-model K optimalizácia (len pre dotaznik/kombinacia)
        if gtype == "anamneza":
            k_opt = None
        else:
            print(f"    {mname:<16}  hladam K_opt ...", end=" ", flush=True)
            k_opt, k_cv_best = najdi_k_opt(base_clf, needs_scale, X_tr, y_tr,
                                            K_GRID, gtype, n_dot, cv5)
            print(f"K_opt={k_opt}  (CV={k_cv_best*100:.1f}%)", flush=True)

        clf_copy   = skbase.clone(base_clf)
        pipe       = _make_pipe(clf_copy, needs_scale, k_opt)
        cv_scores  = cross_val_score(pipe, X_tr, y_tr, cv=cv5, scoring='roc_auc', n_jobs=1)

        # NOVINKA: prah sa volí na validačných foldoch CV (nie na test sete)
        # Zbierame predikcie na validačných foldoch (out-of-fold proba na train)
        _oof_proba = np.zeros(len(y_tr))
        for _tr_f, _val_f in cv5.split(X_tr, y_tr):
            _clf_f = skbase.clone(base_clf)
            _pipe_f = _make_pipe(_clf_f, needs_scale, k_opt, _gtype=gtype)
            _pipe_f.fit(X_tr[_tr_f], y_tr[_tr_f])
            _oof_proba[_val_f] = _pipe_f.predict_proba(X_tr[_val_f])[:, 1]
        # Výber prahu na OOF proba (= validačné foldy, žiadny únik z test setu)
        opt_thr_cv = vyber_prah_na_val(_oof_proba, y_tr)

        clf_final  = skbase.clone(base_clf)
        pipe_final = _make_pipe(clf_final, needs_scale, k_opt)
        pipe_final.fit(X_tr, y_tr)
        proba = pipe_final.predict_proba(X_te)[:,1]
        m = calc_metrics(y_te, proba, thr=opt_thr_cv)

        k_str = f" K={k_opt}" if k_opt is not None else ""
        all_results[gname][mname] = {
            'AUC_test':m['AUC'],'AUC_CV':cv_scores.mean(),'AUC_CV_std':cv_scores.std(ddof=1),
            'cv_scores': cv_scores,
            'Sens':m['Sens'],'Spec':m['Spec'],'F1':m['F1'],'Acc':m['Acc'],
            'TP':m['TP'],'FP':m['FP'],'FN':m['FN'],'TN':m['TN'],
            'proba_te':proba,'pipeline':pipe_final,'k_opt':k_opt,
            'threshold': opt_thr_cv,
        }
        print(f"    {mname:<16}{k_str:<7} AUC_test={m['AUC']*100:.1f}%  "
              f"CV={cv_scores.mean()*100:.1f}+/-{cv_scores.std(ddof=1)*100:.1f}%  "
              f"Prah={opt_thr_cv:.2f}  Sens={m['Sens']*100:.1f}%  Spec={m['Spec']*100:.1f}%")

        rows_csv.append({
            "Skupina":gname,"Model":mname,"K_opt":k_opt,
            "AUC_test_%":round(m['AUC']*100,1),
            "AUC_CV_%":round(cv_scores.mean()*100,1),
            "AUC_CV_std_%":round(cv_scores.std(ddof=1)*100,1),
            "Threshold_CV":round(opt_thr_cv,3),
            "Sens_%":round(m['Sens']*100,1),"Spec_%":round(m['Spec']*100,1),
            "F1_%":round(m['F1']*100,1),"Acc_%":round(m['Acc']*100,1),
            "TP":m['TP'],"FP":m['FP'],"FN":m['FN'],"TN":m['TN']
        })

pd.DataFrame(rows_csv).to_csv(
    os.path.join(OUT_DIR,'vysledky_10f_porovnanie.csv'), index=False)
print(f"\n  Ulozene: vysledky_10f_porovnanie.csv")

# ==============================================================================
# SEKCIA 6: VÝBER KANDIDÁTSKEHO MODELU
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 6: VYBER KANDIDATSKEHO MODELU (automaticky – max AUC_CV)")
print("="*70)

# Výber víťaza:
#   1. Ak existujú nested CV výsledky pre model v Kombinacia skupine → použij nested CV AUC
#      (nestranný odhad, bez model selection bias)
#   2. Pre ostatné skupiny (Anamneza, Dotaznik) → simple CV AUC (nested CV pre ne nebežalo)
#   3. Finálne porovnanie: najlepší Kombinacia (nested) vs. najlepší Anamneza/Dotaznik (simple)

# Najdi víťaza z Kombinacia podľa nested CV
best_kom_nested_auc = -1; best_kom_nested_name = None
for mname in nested_results:
    auc_n = nested_results[mname]['mean_auc']
    if auc_n > best_kom_nested_auc:
        best_kom_nested_auc = auc_n; best_kom_nested_name = mname

# Najdi víťaza z každej skupiny podľa simple CV
best_cv = -1; best_key = None
for gname, mods in all_results.items():
    for mname, r in mods.items():
        if mname == 'Baseline': continue
        if r['AUC_CV'] > best_cv:
            best_cv = r['AUC_CV']; best_key = (gname, mname)

bg_auto, bm_auto = best_key
br_auto = all_results[bg_auto][bm_auto]

# Porovnaj: víťaz Kombinacia (nested) vs. celkový víťaz (simple CV)
if best_kom_nested_name and best_kom_nested_name in all_results.get('Kombinacia', {}):
    br_kom_nested = all_results['Kombinacia'][best_kom_nested_name]
    print(f"\n  Vitaz Kombinacia podla NESTED CV: Kombinacia/{best_kom_nested_name}  "
          f"nested_AUC={best_kom_nested_auc*100:.1f}%  "
          f"simple_AUC={br_kom_nested['AUC_CV']*100:.1f}%")
print(f"  Vitaz podla simple CV (vsetky skupiny): {bg_auto}/{bm_auto}  "
      f"AUC_CV={br_auto['AUC_CV']*100:.1f}%")

# Použi víťaza podľa nested CV z Kombinacia skupiny (metodicky čistejšie)
if best_kom_nested_name and best_kom_nested_name in all_results.get('Kombinacia', {}):
    bg_auto = 'Kombinacia'; bm_auto = best_kom_nested_name
    br_auto = all_results['Kombinacia'][best_kom_nested_name]
    print(f"  → Finalny vitaz (nested CV): Kombinacia/{bm_auto}")
else:
    print(f"  → Finalny vitaz (simple CV fallback): {bg_auto}/{bm_auto}")

bg, bm, br = bg_auto, bm_auto, br_auto
base_r = all_results[bg].get('Baseline')
print(f"\n  KANDIDATSKY FINALNY MODEL: {bg} / {bm}")
print(f"    AUC_CV={br['AUC_CV']*100:.1f}% ±{br['AUC_CV_std']*100:.1f}%  "
      f"AUC_test={br['AUC_test']*100:.1f}%")
print(f"    Threshold (z CV val foldov): {br['threshold']:.2f}")
if base_r:
    print(f"    Prinos nad baseline: +{(br['AUC_CV']-base_r['AUC_CV'])*100:.1f} pp")

proba_best = br['proba_te']
opt_thr    = br['threshold']

# ==============================================================================
# SEKCIA 7: POROVNANIE 10d vs. 10f METODIKY
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 7: POROVNANIE METODIKY 10d vs 10f")
print("="*70)

rf_k = all_results.get("Kombinacia",{}).get("RF",{}).get("k_opt","?")
print(f"""
  10d (predchadzajuca analyza):
    - K_opt per-model cez 5-fold CV
    - Prah vybrany na test sete (optimisticky – test set kontaminovany pre vyber prahu)
    - AUC_CV = vnutorna CV (rovnako optimisticka)

  10f (tato analyza):
    - K_opt per-model cez 5-fold CV (rovnake ako 10d → porovnatelne K)
    - Nested CV (vonkajsi 5-fold × vnutorny 5-fold): NESTRANNÝ odhad AUC
      → prah vybrany na validacnom folde (nie na test sete)
    - Finalny model: prah z OOF proba na train CV (nie z test setu)
    - Test set: VYLUCNE pre finalnu evaluaciu

  RF K_opt={rf_k}  |  Vitaz: {bg}/{bm}
    AUC_CV  = {br['AUC_CV']*100:.1f}% (optimisticky – z 5-fold CV)
    AUC_test = {br['AUC_test']*100:.1f}% (finalny – test set skutocne held-out)
""")

# ==============================================================================
# SEKCIA 8: PREHĽAD PRAHU
# ==============================================================================
print("\n" + "="*70)
print(f"  SEKCIA 8: PREHLADOVA PRAHOVA ANALYZA  (prah={opt_thr:.2f} z CV)")
print("="*70)

thresholds_scan = np.arange(0.10, 0.91, 0.05)
prah_rows = []

print(f"\n  {'Prah':>5}  {'Sens%':>7}  {'Spec%':>7}  {'F1%':>6}  "
      f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}")
print("  " + "-"*55)
for t in thresholds_scan:
    m = calc_metrics(y_te, proba_best, thr=t)
    youden = m['Sens'] + m['Spec'] - 1
    print(f"  {t:>5.2f}  {m['Sens']*100:>7.1f}  {m['Spec']*100:>7.1f}  "
          f"{m['F1']*100:>6.1f}  {m['TP']:>4}  {m['FP']:>4}  {m['FN']:>4}  {m['TN']:>4}")
    prah_rows.append({"Prah":round(t,2),"Sens_%":round(m['Sens']*100,1),
                      "Spec_%":round(m['Spec']*100,1),"F1_%":round(m['F1']*100,1),
                      "TP":m['TP'],"FP":m['FP'],"FN":m['FN'],"TN":m['TN'],
                      "Youden":round(m['Sens']+m['Spec']-1,3)})

print(f"\n  Zvoleny prah (z CV val foldov): {opt_thr:.2f}")
m_opt = calc_metrics(y_te, proba_best, thr=opt_thr)
print(f"  Sens={m_opt['Sens']*100:.1f}%  Spec={m_opt['Spec']*100:.1f}%  "
      f"FN={m_opt['FN']}  FP={m_opt['FP']}")

# ==============================================================================
# SEKCIA 9: BOOTSTRAP CI
# ==============================================================================
print("\n" + "="*70)
print(f"  SEKCIA 9: BOOTSTRAP CI (prah={opt_thr:.2f}, n_boot=1000)")
print("="*70)

ci_rows = []
for mname_m, fn in [
    ("AUC",  lambda yt,ys: roc_auc_score(yt,ys) if len(np.unique(yt))>1 else 0.5),
    ("Sens", lambda yt,ys: calc_metrics(yt,ys,opt_thr)['Sens']),
    ("Spec", lambda yt,ys: calc_metrics(yt,ys,opt_thr)['Spec']),
    ("F1",   lambda yt,ys: calc_metrics(yt,ys,opt_thr)['F1']),
]:
    val = fn(y_te, proba_best)
    lo, hi = bootstrap_ci(y_te, proba_best, fn)
    print(f"  {mname_m:<6}  {val*100:>6.1f}%  [{lo*100:.1f}%–{hi*100:.1f}%]")
    ci_rows.append({"Metrika":mname_m,"Hodnota_%":round(val*100,1),
                    "CI_low_%":round(lo*100,1),"CI_high_%":round(hi*100,1),
                    "Prah":opt_thr,"Model":bm,"Skupina":bg})

pd.DataFrame(ci_rows).to_csv(
    os.path.join(OUT_DIR,'vysledky_10f_bootstrap_ci.csv'), index=False)

# ==============================================================================
# SEKCIA 10: SUHRNA TABULKA (všetky modely)
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 10: SUHRNA TABULKA (zoradene podla AUC_CV)")
print("="*70)

all_rows = []
for gname, mods in all_results.items():
    for mname, r in mods.items():
        if mname=='Baseline': continue
        all_rows.append({
            'Skupina':gname,'Model':mname,
            'AUC_CV':r['AUC_CV'],'AUC_CV_std':r['AUC_CV_std'],'AUC_test':r['AUC_test'],
            'Threshold_CV':r.get('threshold', None),
            'Sens':r['Sens'],'Spec':r['Spec'],
        })

all_df = pd.DataFrame(all_rows).sort_values('AUC_CV', ascending=False)
print(f"\n  {'#':<3} {'Skupina':<12} {'Model':<16} {'AUC_CV':>8} {'±':>5} {'AUC_test':>9}"
      f"  {'Prah':>6} {'Sens%':>6} {'Spec%':>6}")
print("  "+"-"*80)
for i, row in enumerate(all_df.itertuples(),1):
    marker = " ◄" if (row.Skupina==bg and row.Model==bm) else ""
    thr_s = f"{row.Threshold_CV:.2f}" if row.Threshold_CV is not None else "–"
    print(f"  {i:<3} {row.Skupina:<12} {row.Model:<16} "
          f"{row.AUC_CV*100:>7.1f}%  {row.AUC_CV_std*100:>4.1f}%  "
          f"{row.AUC_test*100:>8.1f}%  {thr_s:>6}  "
          f"{row.Sens*100:>5.1f}%  {row.Spec*100:>5.1f}%{marker}")

print(f"\n  TOP 5:")
for i, row in enumerate(all_df.head(5).itertuples(),1):
    marker = " ◄" if (row.Skupina==bg and row.Model==bm) else ""
    print(f"  {i}. {row.Skupina}/{row.Model:<28} "
          f"AUC_CV={row.AUC_CV*100:.1f}%  AUC_test={row.AUC_test*100:.1f}%{marker}")

all_df_out = all_df.copy()
for c in ['AUC_CV','AUC_CV_std','AUC_test','Sens','Spec']:
    all_df_out[c] = (all_df_out[c]*100).round(1)
all_df_out.to_csv(os.path.join(OUT_DIR,'vysledky_10f_vsetky_modely.csv'), index=False)

# ==============================================================================
# SEKCIA 11: KALIBRÁCIA
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 11: KALIBRÁCIA")
print("="*70)

bs = brier_score_loss(y_te, proba_best)
bs_base = brier_score_loss(y_te, np.full(len(y_te), y_te.mean()))
bss = 1 - bs/bs_base
print(f"  Brier score: {bs:.4f}  |  BSS: {bss:.3f}")

try:
    from sklearn.calibration import calibration_curve as sk_calib
    prob_true, prob_pred = sk_calib(y_te, proba_best, n_bins=8, strategy='quantile')
    fig, ax = plt.subplots(figsize=(7,6))
    fig.patch.set_facecolor("#F8F7F5"); ax.set_facecolor("#F8F7F5")
    ax.plot([0,1],[0,1],"k--",lw=1.2,label="Perfectna kalibracia")
    ax.plot(prob_pred, prob_true,"-o",color="#2E86AB",lw=2,
            label=f"{bm} {bg} (Brier={bs:.3f})")
    ax.set_xlabel("Predikovaná pravd."); ax.set_ylabel("Pozorovaná freq.")
    k_label = br.get('k_opt') if bg == 'Kombinacia' else '–'
    ax.set_title(f"Kalibrácia – {bm} ({bg})\n10f: K_DOT={k_label}, prah z CV val foldov",
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'graf_10f_kalibracia.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Ulozeny: graf_10f_kalibracia.png")
except Exception as e:
    print(f"  Graf kalibracie: {e}")

# ==============================================================================
# SEKCIA 11b: EXPORT PIPELINE
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 11b: EXPORT PIPELINE (joblib)")
print("="*70)

model_package = {
    "pipeline":   br['pipeline'], "features": GROUPS[bg]["feats"],
    "threshold":  opt_thr,        "model_name": bm, "group": bg,
    "target":     "A10",          "k_dot": br.get("k_opt") if bg=="Kombinacia" else None,
    "n_ana":      N_ANA,
    "AUC_CV":     round(br['AUC_CV']*100,1),
    "AUC_CV_std": round(br['AUC_CV_std']*100,1),
    "AUC_test":   round(br['AUC_test']*100,1),
    "threshold_note": ("Prah odvodeny z OOF proba na trenovacom sete (nie z test setu). "
                       "Vyžaduje externu validaciu pred klinickym pouzitim."),
}
joblib.dump(model_package, os.path.join(OUT_DIR,"model_10f_kombinacia.joblib"))
print(f"  Ulozeny: model_10f_kombinacia.joblib  ({bm}/{bg})")

# Anamneza model export
ana_r = all_results.get("Anamneza",{}).get(APP_ANA_MODEL)
if ana_r:
    # Pre Anamneza: prah z OOF
    _X_tr_ana = df_valid[ANAMNEZA].values[tr_idx].astype(float)
    _oof_ana = np.zeros(len(y_tr))
    for _tr_f, _val_f in cv5.split(_X_tr_ana, y_tr):
        _clf_f = skbase.clone(
            dict([(m,c) for m,c,_ in MODEL_DEFS if m==APP_ANA_MODEL])[APP_ANA_MODEL]
        )
        _pipe_f = make_pipeline_anamneza(_clf_f)
        _pipe_f.fit(_X_tr_ana[_tr_f], y_tr[_tr_f])
        _oof_ana[_val_f] = _pipe_f.predict_proba(_X_tr_ana[_val_f])[:, 1]
    ana_thr = vyber_prah_na_val(_oof_ana, y_tr)
    joblib.dump({
        "pipeline":   ana_r['pipeline'], "features": GROUPS["Anamneza"]["feats"],
        "threshold":  ana_thr,           "model_name": APP_ANA_MODEL, "group": "Anamneza",
        "target":     "A10",
        "AUC_CV":     round(ana_r['AUC_CV']*100,1),
        "AUC_CV_std": round(ana_r['AUC_CV_std']*100,1),
        "AUC_test":   round(ana_r['AUC_test']*100,1),
        "threshold_note": ("Prah odvodeny z OOF proba na trenovacom sete (nie z test setu). "
                           "Vyžaduje externu validaciu pred klinickym pouzitim."),
    }, os.path.join(OUT_DIR,"model_10f_anamneza.joblib"))
    print(f"  Ulozeny: model_10f_anamneza.joblib  "
          f"(Anamneza/{APP_ANA_MODEL}, 5 features, prah={ana_thr:.2f})")

# ==============================================================================
# SEKCIA 11x: DECISION CURVE ANALYSIS (DCA)
# ==============================================================================
# DCA: pre každý pravdepodobnostný prah t ∈ [0.01, 0.99]:
#   net_benefit(t) = (TP/n) - (FP/n) * (t / (1-t))
#   treat_all(t)   = prevalence - (1-prevalence) * (t / (1-t))
#   treat_none     = 0 (vodorovná čiara)
#
# Klinická interpretácia: net benefit vyjadruje pomer pravdivých pozitívov
# vzhľadom na falošne pozitívnych, vážený klinickým rozhodovacím prahom t.
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 11x: DECISION CURVE ANALYSIS (DCA)")
print("  Model: Kombinacia/RF  |  Test set (20%)")
print("="*70)

try:
    dca_t_range = np.linspace(0.01, 0.99, 199)
    n_te = len(y_te)
    prevalence = y_te.mean()

    # Vypočítaj net benefit pre model (Kombinacia/RF)
    kom_rf_r = all_results.get("Kombinacia", {}).get(APP_KOM_MODEL)
    if kom_rf_r is None:
        # Fallback: použi víťazný model
        kom_rf_r = br

    proba_dca = kom_rf_r['proba_te']

    nb_model     = []
    nb_treat_all = []
    nb_treat_none= []

    for t in dca_t_range:
        # Model
        pred_t = (proba_dca >= t).astype(int)
        TP_t = int(((pred_t==1)&(y_te==1)).sum())
        FP_t = int(((pred_t==1)&(y_te==0)).sum())
        nb_m = (TP_t / n_te) - (FP_t / n_te) * (t / (1.0 - t))
        nb_model.append(nb_m)

        # Treat all
        nb_ta = prevalence - (1.0 - prevalence) * (t / (1.0 - t))
        nb_treat_all.append(nb_ta)

        # Treat none = 0
        nb_treat_none.append(0.0)

    nb_model     = np.array(nb_model)
    nb_treat_all = np.array(nb_treat_all)

    # Uloženie DCA dát
    dca_df = pd.DataFrame({
        'Threshold': np.round(dca_t_range, 4),
        'NB_Model':  np.round(nb_model,    5),
        'NB_Treat_All': np.round(nb_treat_all, 5),
        'NB_Treat_None': 0.0,
    })
    dca_df.to_csv(os.path.join(OUT_DIR, 'vysledky_10f_dca.csv'), index=False)
    print(f"  Ulozene: vysledky_10f_dca.csv  ({len(dca_df)} riadkov)")

    # Graf DCA
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F7F5"); ax.set_facecolor("#F8F7F5")

    ax.plot(dca_t_range, nb_model, color='#2E86AB', lw=2.5,
            label=f'Model: Kombinacia/{APP_KOM_MODEL}')
    ax.plot(dca_t_range, np.maximum(nb_treat_all, -0.05), color='#e74c3c', lw=1.8,
            linestyle='--', label='Treat all')
    ax.axhline(y=0, color='#555', lw=1.5, linestyle=':', label='Treat none')

    # Zvýrazni oblasť kde model prevyšuje obe referenčné stratégie
    _clip_ta = np.maximum(nb_treat_all, 0)
    _model_best_mask = (nb_model > _clip_ta) & (nb_model > 0)
    if _model_best_mask.any():
        _t_start = dca_t_range[_model_best_mask][0]
        _t_end   = dca_t_range[_model_best_mask][-1]
        ax.axvspan(_t_start, _t_end, alpha=0.12, color='#2E86AB',
                   label=f'Model dominuje ({_t_start:.2f}–{_t_end:.2f})')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, max(prevalence + 0.05, nb_model.max() + 0.02))
    ax.set_xlabel("Pravdepodobnostný prah (t)", fontsize=12)
    ax.set_ylabel("Čistý prínos (Net Benefit)", fontsize=12)
    ax.set_title(f"Decision Curve Analysis – Kombinacia/{APP_KOM_MODEL}\n"
                 f"n_test={n_te}, prevalencia={prevalence*100:.1f}%, "
                 f"prah z CV val foldov={opt_thr:.2f}",
                 fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'graf_10f_dca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Ulozeny: graf_10f_dca.png")

    # Vypíš kľúčové body DCA
    print(f"\n  DCA kľúčové body (Model vs Treat-all):")
    print(f"  {'t':>5}  {'NB_model':>10}  {'NB_treatall':>12}  {'Lepsi':>8}")
    print("  " + "-"*45)
    for t_show in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        idx_t = np.argmin(np.abs(dca_t_range - t_show))
        nb_m_show  = nb_model[idx_t]
        nb_ta_show = nb_treat_all[idx_t]
        better = "Model" if nb_m_show > max(nb_ta_show, 0) else ("Treat-all" if nb_ta_show > 0 else "Treat-none")
        print(f"  {t_show:>5.2f}  {nb_m_show:>10.4f}  {nb_ta_show:>12.4f}  {better:>8}")

except Exception as e:
    print(f"  DCA: {e}")
    import traceback; traceback.print_exc()

# ==============================================================================
# SEKCIA 11y: POROVNANIE 10d vs 10f (side-by-side)
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 11y: POROVNANIE 10d vs 10f SIDE-BY-SIDE")
print("="*70)

print(f"""
  Metodologické porovnanie – Kombinacia/{APP_KOM_MODEL}:

  Metrika            │ 10d (optimisticky)          │ 10f (tento skript)
  ───────────────────┼─────────────────────────────┼────────────────────────────
  AUC_CV             │ z 5-fold CV (biased)         │ z Nested CV (unbiased)
  Výber K            │ CV na celom train sete       │ vnútorný CV per outer fold
  Výber prahu        │ na test sete (!)             │ na OOF val foldoch (correct)
  AUC_test           │ optimistický (prah z test)   │ skutočne held-out test set
  DCA                │ nie                          │ áno (sekcia 11x)
  ───────────────────┴─────────────────────────────┴────────────────────────────

  Aktuálne výsledky 10f:
    Nested CV AUC (key modely):""")

for mname in NESTED_CV_MODELS:
    if mname in nested_results:
        res = nested_results[mname]
        print(f"      {mname:<16}  AUC_nested={res['mean_auc']*100:.1f}% ± "
              f"{res['std_auc']*100:.1f}%  "
              f"Sens={res['mean_sens']*100:.1f}%  "
              f"Spec={res['mean_spec']*100:.1f}%")

rf_r = all_results.get("Kombinacia", {}).get(APP_KOM_MODEL)
if rf_r:
    print(f"\n    {APP_KOM_MODEL}/Kombinacia finalne (10f):")
    print(f"      AUC_CV   = {rf_r['AUC_CV']*100:.1f}% ± {rf_r['AUC_CV_std']*100:.1f}%"
          f"  (5-fold CV, optimisticky)")
    print(f"      AUC_test = {rf_r['AUC_test']*100:.1f}%  (held-out 20%, prah={rf_r['threshold']:.2f})")
    rf_nested = nested_results.get(APP_KOM_MODEL)
    if rf_nested:
        print(f"      AUC_nest = {rf_nested['mean_auc']*100:.1f}% ± "
              f"{rf_nested['std_auc']*100:.1f}%  (nested CV, unbiased)")

# ==============================================================================
# SEKCIA 11z: ROC KRIVKY
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 11z: ROC KRIVKY")
print("="*70)

try:
    from sklearn.metrics import roc_curve, auc as _auc

    def _best_model_in_group(group_name, preferred):
        _r = all_results.get(group_name, {}).get(preferred)
        if _r is not None:
            return preferred, _r
        _best_m, _best_r = None, None
        for _m, _res in all_results.get(group_name, {}).items():
            if _best_r is None or _res.get('AUC_CV', 0) > _best_r.get('AUC_CV', 0):
                _best_m, _best_r = _m, _res
        return _best_m, _best_r

    _dot_preferred = APP_KOM_MODEL if APP_KOM_MODEL in all_results.get("Dotaznik", {}) else "ExtraTrees"
    _roc_models = {
        "Anamneza":   _best_model_in_group("Anamneza",   APP_ANA_MODEL),
        "Dotaznik":   _best_model_in_group("Dotaznik",   _dot_preferred),
        "Kombinacia": _best_model_in_group("Kombinacia", APP_KOM_MODEL),
    }
    _grp_colors = {"Anamneza": "#3498db", "Dotaznik": "#e67e22", "Kombinacia": "#27ae60"}

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#F8F7F5"); ax.set_facecolor("#F8F7F5")
    ax.plot([0,1],[0,1],'k--', lw=1, label='Náhodný klasifikátor (AUC=0.500)')

    for _gname in ["Anamneza", "Dotaznik", "Kombinacia"]:
        _mname, _r = _roc_models[_gname]
        if _r is None: continue
        _fpr, _tpr, _ = roc_curve(y_te, _r['proba_te'])
        _roc_auc = _auc(_fpr, _tpr)
        _lw = 2.5 if _gname == "Kombinacia" else 1.8
        ax.plot(_fpr, _tpr, lw=_lw, color=_grp_colors[_gname],
                label=f"{_gname}/{_mname}  (AUC={_roc_auc:.3f})")
        print(f"  {_gname}/{_mname}: AUC={_roc_auc:.3f}")

    ax.set_xlabel("1 – Špecificita (FPR)", fontsize=11)
    ax.set_ylabel("Senzitivita (TPR)", fontsize=11)
    ax.set_title(f"ROC krivky – porovnanie skupín\n(n_test={len(y_te)}, seed={SEED})",
                 fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'graf_10f_roc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Ulozeny: graf_10f_roc.png")
except Exception as e:
    print(f"  ROC krivky: {e}")

# ==============================================================================
# SEKCIA 12: SHAP – INTERPRETOVATEĽNOSŤ MODELU
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 12: SHAP – interpretovateľnosť finálneho modelu")
print(f"  Model: {bg}/{bm}")
print("="*70)

try:
    import shap

    _pipe_shap = br['pipeline']
    _feats_shap = GROUPS[bg]["feats"]
    _X_te_shap = df_valid[_feats_shap].iloc[te_idx].values.astype(float)

    # Získaj transformovaný X po imputácii a selekcii
    _X_te_transformed = _pipe_shap[:-1].transform(_X_te_shap)

    # Získaj samotný klasifikátor (bez CalibratedClassifierCV obaľovača)
    _clf_inner = _pipe_shap.named_steps['clf']

    # Skús TreeExplainer (RF, ExtraTrees, GradBoost...)
    try:
        if hasattr(_clf_inner, 'estimators_'):
            # CalibratedClassifierCV – použi base estimator
            _base_est = _clf_inner.estimators_[0].estimator if hasattr(_clf_inner, 'estimators_') else _clf_inner
            _explainer = shap.TreeExplainer(_base_est)
        else:
            _explainer = shap.TreeExplainer(_clf_inner)
        _shap_vals = _explainer.shap_values(_X_te_transformed)
        # Pre binárnu klasifikáciu: shap_values môže byť list [neg, pos]
        if isinstance(_shap_vals, list):
            _shap_vals = _shap_vals[1]
    except Exception:
        # Fallback: KernelExplainer (pomalší, univerzálny)
        _explainer = shap.KernelExplainer(
            lambda x: _pipe_shap.predict_proba(
                np.hstack([np.zeros((len(x), len(_feats_shap) - _X_te_transformed.shape[1])), x])
            )[:, 1],
            shap.kmeans(_X_te_transformed, 10)
        )
        _shap_vals = _explainer.shap_values(_X_te_transformed)

    # Názvy features po ColumnTransformer selekcii
    if bg == 'Kombinacia':
        _ct = _pipe_shap.named_steps['selector']
        _shap_feat_names = []
        for _tname, _trans, _cols in _ct.transformers_:
            if _tname == 'ana':
                _shap_feat_names += [_feats_shap[c] for c in _cols]
            elif _tname == 'dot' and hasattr(_trans, 'get_support'):
                _dot_feats_all = [_feats_shap[c] for c in _cols]
                _shap_feat_names += [f for f, m in zip(_dot_feats_all, _trans.get_support()) if m]
    else:
        _shap_feat_names = _feats_shap

    # Bar plot (mean |SHAP|)
    _mean_abs_shap = np.abs(_shap_vals).mean(axis=0)
    _shap_df = pd.DataFrame({'Feature': _shap_feat_names[:len(_mean_abs_shap)],
                             'Mean_SHAP': _mean_abs_shap}).sort_values('Mean_SHAP', ascending=False)
    _shap_df.to_csv(os.path.join(OUT_DIR, 'vysledky_10f_shap.csv'), index=False)

    fig, ax = plt.subplots(figsize=(9, max(5, len(_shap_df)*0.35)))
    fig.patch.set_facecolor("#F8F7F5"); ax.set_facecolor("#F8F7F5")
    _top_n = min(20, len(_shap_df))
    _plot_df = _shap_df.head(_top_n).iloc[::-1]
    ax.barh(_plot_df['Feature'], _plot_df['Mean_SHAP'], color='#2E86AB', alpha=0.8)
    ax.set_xlabel("Priemerná |SHAP hodnota| (dopad na výstup modelu)", fontsize=10)
    ax.set_title(f"SHAP – Top {_top_n} najdôležitejších premenných\n"
                 f"Model: {bg}/{bm}  |  n_test={len(y_te)}", fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'graf_10f_shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Ulozeny: graf_10f_shap_bar.png")

    # Beeswarm plot
    try:
        fig, ax = plt.subplots(figsize=(10, max(5, _top_n*0.4)))
        fig.patch.set_facecolor("#F8F7F5")
        shap.summary_plot(_shap_vals[:, :_top_n],
                         _X_te_transformed[:, :_top_n],
                         feature_names=_shap_feat_names[:_top_n],
                         show=False, plot_size=None)
        plt.title(f"SHAP Beeswarm – {bg}/{bm}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'graf_10f_shap_beeswarm.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Ulozeny: graf_10f_shap_beeswarm.png")
    except Exception as e_bee:
        print(f"  Beeswarm: {e_bee}")

    print(f"\n  Top 10 features podla SHAP:")
    for _, row in _shap_df.head(10).iterrows():
        print(f"    {row['Feature']:<25}  mean|SHAP|={row['Mean_SHAP']:.4f}")

except ImportError:
    print("  SHAP nie je nainštalovaný. Spusti: pip install shap")
except Exception as e:
    print(f"  SHAP chyba: {e}")
    import traceback; traceback.print_exc()

# ==============================================================================
# SEKCIA 13: KORELÁCIE VSTUPNÝCH PREMENNÝCH
# ==============================================================================
print("\n" + "="*70)
print("  SEKCIA 13: KORELÁCIE VSTUPNÝCH PREMENNÝCH (selected features)")
print("="*70)

try:
    _feats_cor = GROUPS['Kombinacia']['feats']
    _X_cor = df_valid[_feats_cor].iloc[tr_idx].values.astype(float)
    _X_cor_imp = SimpleImputer(strategy='median').fit_transform(_X_cor)
    _cor_df = pd.DataFrame(_X_cor_imp, columns=_feats_cor)

    # Vyber len selected_dot + anamneza features
    _sel_feats_cor = ANAMNEZA + (br.get('pipeline').named_steps['selector']
                                  .transformers_[1][1].get_feature_names_out().tolist()
                                  if bg == 'Kombinacia' and hasattr(
                                      br.get('pipeline').named_steps.get('selector', object()),
                                      'transformers_') else DOTAZNIK_KAND[:30])
    _sel_feats_cor = [f for f in _sel_feats_cor if f in _cor_df.columns][:35]
    _cor_matrix = _cor_df[_sel_feats_cor].corr()

    # Uloženie
    _cor_matrix.to_csv(os.path.join(OUT_DIR, 'vysledky_10f_korelacie.csv'))
    print(f"  Ulozene: vysledky_10f_korelacie.csv ({len(_sel_feats_cor)} features)")

    # Heatmapa
    _n_f = len(_sel_feats_cor)
    fig, ax = plt.subplots(figsize=(max(10, _n_f*0.5), max(8, _n_f*0.5)))
    fig.patch.set_facecolor("#F8F7F5")
    _cm = ax.imshow(_cor_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(_cm, ax=ax, label='Pearsonov korelačný koeficient')
    ax.set_xticks(range(_n_f)); ax.set_xticklabels(_sel_feats_cor, rotation=90, fontsize=7)
    ax.set_yticks(range(_n_f)); ax.set_yticklabels(_sel_feats_cor, fontsize=7)
    ax.set_title(f"Korelácie vstupných premenných – {bg}/{bm}\n"
                 f"(trénovacia sada, n={len(tr_idx)})", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'graf_10f_korelacie.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Ulozeny: graf_10f_korelacie.png")

    # Vypíš najvyššie korelácie (okrem diagonály)
    _cor_pairs = []
    for i in range(len(_sel_feats_cor)):
        for j in range(i+1, len(_sel_feats_cor)):
            _cor_pairs.append((_sel_feats_cor[i], _sel_feats_cor[j],
                               abs(_cor_matrix.iloc[i, j])))
    _cor_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top 10 najvyššich korelacii (|r|):")
    for f1, f2, r in _cor_pairs[:10]:
        print(f"    {f1:<20} × {f2:<20}  |r|={r:.3f}")

except Exception as e:
    print(f"  Korelacie chyba: {e}")
    import traceback; traceback.print_exc()

# ==============================================================================
# ZÁVER
# ==============================================================================
print("\n" + "="*70)
print("  ZÁVER: 10f_hutt_predikcia.py – DOKONCENE")
print("="*70)

print(f"""
  Výstupné súbory:
    vysledky_10f_nested_cv.csv    – výsledky nested CV (všetky modely × 5 foldov)
    vysledky_10f_wilcoxon.csv     – Wilcoxon test porovnania top modelov
    vysledky_10f_porovnanie.csv   – finálne porovnanie (20 modelov × 3 skupiny)
    vysledky_10f_vsetky_modely.csv– zoradená tabuľka všetkých modelov
    vysledky_10f_bootstrap_ci.csv – bootstrap 95% CI finálneho modelu
    vysledky_10f_dca.csv          – Decision Curve Analysis dáta
    vysledky_10f_shap.csv         – SHAP hodnoty (mean |SHAP| per feature)
    vysledky_10f_korelacie.csv    – korelačná matica selected features
    graf_10f_nested_cv.png        – boxplot AUC cez vonkajšie foldy (všetky modely)
    graf_10f_dca.png              – DCA krivka (net benefit vs. prah)
    graf_10f_roc.png              – ROC krivky (3 skupiny)
    graf_10f_kalibracia.png       – kalibračná krivka
    graf_10f_shap_bar.png         – SHAP bar plot (top 20 features)
    graf_10f_shap_beeswarm.png    – SHAP beeswarm plot
    graf_10f_korelacie.png        – heatmapa korelácií
    model_10f_kombinacia.joblib   – finálny pipeline ({bg}/{bm}, nested CV víťaz)
    model_10f_anamneza.joblib     – anamnéza pipeline (Anamneza/{APP_ANA_MODEL})

  Kľúčové metodologické prínosy 10f:
    1. Nested CV pre VŠETKÝCH 20 modelov → nestranný odhad, bez model selection bias
    2. Víťaz vybraný podľa nested CV AUC (nie simple CV)
    3. Prah z OOF val foldov → test set skutočne held-out pre finálnu evaluáciu
    4. DCA → klinická relevancia modelu cez spektrum rizikových prahov
    5. SHAP → interpretovateľnosť: ktoré premenné model používa a ako
    6. Korelácie → transparentnosť vstupných premenných
    7. Wilcoxon test → štatistické porovnanie top modelov
    8. class_weight='balanced' pre LR a SVM → férovejšie pri miernej nevyváženosti
""")
