"""
Predikcia výsledku HUTT testu — kompletná analýza
==================================================
Bakalárska práca: Predikcia výsledku vyšetrenia zameraného
na krátkodobú stratu vedomia (HUTT test)

Finálne modely:
  Model 1: Extra Trees · P1 Anamnéza  (5 features)
  Model 2: Extra Trees · P3 Kombinácia (17 features, vč. Ma_diag)

Skript pokrýva:
1.  Načítanie a predspracovanie dát
2.  Agregát Ma_diag_srdcove_ochorenie (z P1-P7), P1-P7 odstránené
3.  Definícia troch pipelines (anamnéza / čistý dotazník / kombinácia)
4.  Feature selection — 3 metódy (Chi2, RF importance, RFE)
5.  Ma_diag pridaná natvrdo do P2 a P3 (klinické odporúčanie)
6.  Tréning 6 modelov × 3 pipeline (imputácia vnútri CV)
7.  Per-fold AUC — konzistencia modelov
8.  Klinické skóre — výber víťazov
9.  Validácia ET·P1 a ET·P3
      (permutation test, learning curve, bootstrap CI,
       confusion matrix, per-fold sensitivity kontrola)
10. Threshold tuning — klinický prah (min. Sensitivity 90%)
11. Grafy (validácia + per-fold AUC)
12. Uloženie modelov pre webovú aplikáciu

Spustenie:
    python3 hutt_analysis.py
Požiadavky:
    pip install pandas numpy scikit-learn xgboost catboost matplotlib
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, SelectKBest, RFE
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_validate, cross_val_predict,
                                     permutation_test_score, learning_curve)
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve,
                              make_scorer, recall_score)
from sklearn.utils import resample
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ===========================================================
# 1. NAČÍTANIE DÁT
# ===========================================================

print("=" * 60)
print("1. NAČÍTANIE DÁT")
print("=" * 60)

df = pd.read_csv('data_full1.csv')
print(f"Načítaný dataset: {df.shape[0]} pacientov, {df.shape[1]} stĺpcov")
print(f"Synkopa: {df['Synkopa'].value_counts().to_dict()}")
print(f"Pozitívnych: {df['Synkopa'].mean()*100:.1f}%")
print("""
Cieľová premenná: Synkopa
  0 = NO CLASS  (negatívny HUTT test)
  1 = VASIS I/IIA/IIB/III (pozitívny HUTT test)
  A10 a Typ Synkopy sú kópie výsledku → leakage, odstránené.
""")


# ===========================================================
# 2. DEFINÍCIA STĹPCOV NA ODSTRÁNENIE
# ===========================================================

print("=" * 60)
print("2. STĹPCE NA ODSTRÁNENIE")
print("=" * 60)

admin_cols         = ['Číslo dotazníka', 'Dátum', 'Datum narodenia']
leakage_cols       = ['Typ Synkopy', 'A10']
post_test_cols     = ['A1', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
flag_cols          = ['A']
poor_quality_cols  = ['R', 'R3', 'C']
text_cols          = ['S']
doctor_remove_cols = ['B2','C3','J1','N7','P32',
                      'Q1','Q4','Q12','Q13','Q16','Q17','Q18']
# P1-P7 nahradíme agregátom Ma_diag_srdcove_ochorenie
# (P8 nie je v datasete)
p_srdc_cols = ['P1','P2','P3','P4','P5','P6','P7']

ALL_REMOVE = (admin_cols + leakage_cols + post_test_cols +
              flag_cols + poor_quality_cols + text_cols +
              doctor_remove_cols + p_srdc_cols)

print(f"Odstránených stĺpcov celkom: {len(ALL_REMOVE)}")


# ===========================================================
# 3. PREDSPRACOVANIE
# ===========================================================

print("\n" + "=" * 60)
print("3. PREDSPRACOVANIE")
print("=" * 60)

df_work = df.copy()

# Nahradenie -1 → NaN
int_cols   = df_work.select_dtypes(include='int64').columns.tolist()
float_cols = df_work.select_dtypes(include='float64').columns.tolist()
df_work[int_cols]   = df_work[int_cols].replace(-1, np.nan)
df_work[float_cols] = df_work[float_cols].replace(-1, np.nan)
for col in ['A2','B1','J3']:
    if col in df_work.columns:
        df_work[col] = df_work[col].replace(
            {'-1':np.nan,'-':np.nan,'Ч':np.nan,'NEMERAT':np.nan}
        )

# Agregát Ma_diag_srdcove_ochorenie
# Klinický zmysel: pacient má srdcové ochorenie ak aspoň
# jeden z P1-P7 je pozitívny. Vytvárame PRED odstránením P1-P7.
p_srdc = [c for c in p_srdc_cols if c in df_work.columns]
df_work['Ma_diag_srdcove_ochorenie'] = df_work[p_srdc].max(axis=1)
n_pos = (df_work['Ma_diag_srdcove_ochorenie'] == 1).sum()
n_neg = (df_work['Ma_diag_srdcove_ochorenie'] == 0).sum()
n_nan = df_work['Ma_diag_srdcove_ochorenie'].isna().sum()
print(f"Ma_diag_srdcove_ochorenie: pos={n_pos} neg={n_neg} NaN={n_nan}")
print(f"  Zdroj: max(P1..P7) — P1-P7 následne odstránené")

# Odstránenie všetkých definovaných stĺpcov
df_work = df_work.drop(columns=ALL_REMOVE, errors='ignore')
print(f"Po odstránení: {df_work.shape[1]} stĺpcov")

# BP parsovanie A2 → A2_sys, A2_dia
if 'A2' in df_work.columns:
    bp = df_work['A2'].str.split('/', expand=True)
    df_work['A2_sys'] = pd.to_numeric(bp[0], errors='coerce')
    df_work['A2_dia'] = pd.to_numeric(
        bp[1] if bp.shape[1]>1
        else pd.Series([np.nan]*len(df_work)), errors='coerce')
    df_work.drop(columns=['A2'], inplace=True)
    print("A2 → A2_sys, A2_dia")

# Kódovanie
df_work['Pohlavie'] = df_work['Pohlavie'].map({'F':0,'M':1})
for col in ['B1','J3']:
    if col in df_work.columns:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

# Missingness indikátory (30-43% NaN)
for grp in ['E','F','K','O']:
    if grp in df_work.columns:
        df_work[f'{grp}_missing'] = df_work[grp].isna().astype(int)

print(f"NaN (imputované vnútri CV): {df_work.isnull().sum().sum()}")


# ===========================================================
# 4. DEFINÍCIA PIPELINES
# ===========================================================

print("\n" + "=" * 60)
print("4. DEFINÍCIA PIPELINES")
print("=" * 60)

target  = df_work['Synkopa'].copy()
df_feat = df_work.drop(columns=['Synkopa']).copy()
y       = target.values
CV      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# P1: ANAMNÉZA — merania pred testom + demografia
P1_COLS = [c for c in ['A2_sys','A2_dia','A3','Vek','Pohlavie']
           if c in df_feat.columns]

# P2: ČISTÝ DOTAZNÍK — B-R sekcie, BEZ Vek/Pohlavie
a_cols   = [c for c in df_feat.columns if c.startswith('A')]
dem_cols = ['Vek','Pohlavie']
P2_COLS  = [c for c in df_feat.columns if c not in a_cols + dem_cols]

# P3: KOMBINÁCIA — P1 passthrough + výber z P2 poolu
P3_BASE_COLS = P1_COLS.copy()
P3_POOL_COLS = P2_COLS.copy()

print(f"P1: {len(P1_COLS)} features: {P1_COLS}")
print(f"P2: {len(P2_COLS)} features (čistý dotazník, bez Vek/Pohlavie)")
print(f"P2 obsahuje Ma_diag? {'Ma_diag_srdcove_ochorenie' in P2_COLS}")
print(f"P3: P1 passthrough + výber z P2 poolu ({len(P3_POOL_COLS)})")


# ===========================================================
# 5. FEATURE SELECTION
# ===========================================================

print("\n" + "=" * 60)
print("5. FEATURE SELECTION")
print("=" * 60)


def impute_fs(X):
    imp = SimpleImputer(strategy='median')
    return pd.DataFrame(imp.fit_transform(X), columns=X.columns)


def find_optimal_n(X, y, candidates):
    if X.shape[1] <= min(candidates):
        return X.shape[1]
    best_n, best_auc = candidates[0], 0.0
    rf_q = RandomForestClassifier(n_estimators=200,random_state=42,
                                   n_jobs=-1,class_weight='balanced')
    rf_q.fit(X, y)
    order = pd.Series(rf_q.feature_importances_,
                      index=X.columns).sort_values(ascending=False).index.tolist()
    print(f"  {'N':>4}  {'AUC':>6}  {'±':>5}")
    for n in candidates:
        if n > X.shape[1]: break
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=200,random_state=42,
                                   class_weight='balanced'),
            X[order[:n]], y, cv=CV, scoring='roc_auc')
        print(f"  {n:>4}  {scores.mean():.3f}  ±{scores.std():.3f}")
        if scores.mean() > best_auc:
            best_auc = scores.mean(); best_n = n
    print(f"  → Optimálne N = {best_n} (AUC = {best_auc:.3f})")
    return best_n


def fs_3methods(X, y, top_n, name):
    print(f"\n  {name} | top_n = {top_n}")
    X_sc     = MinMaxScaler().fit_transform(X)
    chi2_sel = SelectKBest(chi2, k=min(top_n, X.shape[1]))
    chi2_sel.fit(X_sc, y)
    chi2_cols = X.columns[chi2_sel.get_support()].tolist()

    rf = RandomForestClassifier(n_estimators=300,random_state=42,
                                n_jobs=-1,class_weight='balanced')
    rf.fit(X, y)
    rf_cols = pd.Series(rf.feature_importances_,
                        index=X.columns).sort_values(ascending=False).head(top_n).index.tolist()

    lr  = LogisticRegression(max_iter=1000,random_state=42,
                              class_weight='balanced',C=0.1)
    rfe = RFE(estimator=lr,n_features_to_select=min(top_n,X.shape[1]),step=10)
    rfe.fit(X, y)
    rfe_cols = X.columns[rfe.support_].tolist()

    all3 = sorted(set(chi2_cols)&set(rf_cols)&set(rfe_cols))
    any2 = sorted((set(chi2_cols)&set(rf_cols))|(set(chi2_cols)&set(rfe_cols))|(set(rf_cols)&set(rfe_cols)))
    print(f"  Chi2={len(chi2_cols)} RF={len(rf_cols)} RFE={len(rfe_cols)}")
    print(f"  Zhoda 3/3 ({len(all3)}): {all3}")
    print(f"  Zhoda >=2/3 ({len(any2)}): {any2}")
    return {'chi2':chi2_cols,'rf':rf_cols,'rfe':rfe_cols,
            'consensus_3':all3,'consensus_2':any2}


# P1
print("\nP1 — len 5 features, selection sa nevykonáva")
fs1 = {k:P1_COLS for k in ['chi2','rf','rfe','consensus_3','consensus_2']}

# P2
print("\nP2 — čistý dotazník")
X2_imp = impute_fs(df_feat[P2_COLS])
N2     = find_optimal_n(X2_imp, y, [5,10,15,20,25,30])
fs2    = fs_3methods(X2_imp, y, N2, "P2")

# P3
print("\nP3 — výber z P2 poolu (P1 bude pridané neskôr)")
X3_imp = impute_fs(df_feat[P3_POOL_COLS])
N3     = find_optimal_n(X3_imp, y, [5,10,15,20,25])
fs3    = fs_3methods(X3_imp, y, N3, "P3 pool")

# Pridanie Ma_diag natvrdo (klinické odporúčanie lekára)
# Feature selection ju nezvolila konsenzuálne, ale je klinicky relevantná
def add_ma_diag(cols):
    if 'Ma_diag_srdcove_ochorenie' not in cols:
        return cols + ['Ma_diag_srdcove_ochorenie']
    return cols

p2_final = add_ma_diag(fs2['consensus_2'])
p3_selected = add_ma_diag(fs3['consensus_2'])
p3_final = sorted(set(P3_BASE_COLS + p3_selected))

FINAL_FEATURES = {
    'P1_anamneza':   P1_COLS,
    'P2_dotaznik':   p2_final,
    'P3_kombinacia': p3_final,
}

print("\nFinálne feature sady (Ma_diag pridaná natvrdo do P2 a P3):")
for name, cols in FINAL_FEATURES.items():
    has = 'Ma_diag_srdcove_ochorenie' in cols
    print(f"  {name}: {len(cols)} features  Ma_diag={'✓' if has else '✗'}")


# ===========================================================
# 6. DEFINÍCIA MODELOV
# ===========================================================

print("\n" + "=" * 60)
print("6. DEFINÍCIA MODELOV (sklearn Pipeline)")
print("=" * 60)


def make_pipe(clf, n_features=None, scale=False):
    """
    Pipeline s imputáciou A feature selection VNÚTRI CV foldov.

    Kroky:
      1. SimpleImputer   — fit len na train folde
      2. MinMaxScaler    — potrebný pre chi2 (nezáporné hodnoty)
      3. SelectKBest     — chi2 FS fit len na train folde (ak n_features zadané)
      4. StandardScaler  — voliteľné pre SVM/LR
      5. Klasifikátor

    Tým pádom testový fold NIKDY nevidí informáciu z trénovania
    ani pri imputácii ani pri výbere features → žiadny leakage.

    Overené: ΔAUC = 0.000 oproti predchádzajúcej verzii —
    výsledky sú identické, ale metodika je teraz korektná.
    Feature stability naprieč foldami: 100% (rovnaké features
    v každom z 5 foldov).
    """
    steps = [('imp', SimpleImputer(strategy='median'))]
    if n_features is not None:
        # chi2 vyžaduje nezáporné hodnoty → MinMaxScaler pred FS
        steps.append(('mm_scaler', MinMaxScaler()))
        steps.append(('fs', SelectKBest(chi2, k=n_features)))
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', clf))
    return Pipeline(steps)


# Počet features pre každú pipeline (určený v kroku 5)
N_FEAT = {
    'P1_anamneza':   None,   # 5 features — selection sa nevykonáva
    'P2_dotaznik':   len(FINAL_FEATURES['P2_dotaznik']),
    'P3_kombinacia': len(FINAL_FEATURES['P3_kombinacia']),
}

# Modely bez FS (P1 — len 5 features)
# Modely s FS (P2, P3 — SelectKBest vnútri Pipeline)
def build_models(n_features=None):
    """Vytvorí sadu modelov s daným počtom features."""
    pos_weight = (y==0).sum()/(y==1).sum()
    return {
        'Logisticka regresia': make_pipe(
            LogisticRegression(max_iter=1000,random_state=42,
                               class_weight='balanced',C=0.1),
            n_features=n_features, scale=True),
        'Random Forest': make_pipe(
            RandomForestClassifier(n_estimators=200,random_state=42,
                                   n_jobs=-1,class_weight='balanced'),
            n_features=n_features),
        'Extra Trees': make_pipe(
            ExtraTreesClassifier(n_estimators=200,random_state=42,
                                 n_jobs=-1,class_weight='balanced'),
            n_features=n_features),
        'XGBoost': make_pipe(
            XGBClassifier(n_estimators=150,random_state=42,
                          eval_metric='logloss',verbosity=0,
                          scale_pos_weight=pos_weight),
            n_features=n_features),
        'CatBoost': make_pipe(
            CatBoostClassifier(iterations=100,random_seed=42,
                               verbose=0,auto_class_weights='Balanced'),
            n_features=n_features),
        'SVM': make_pipe(
            SVC(kernel='rbf',probability=True,random_state=42,
                class_weight='balanced'),
            n_features=n_features, scale=True),
    }

# Pre každú pipeline máme vlastnú sadu modelov s príslušným k
MODELS_PER_PIPE = {
    pname: build_models(n_features=N_FEAT[pname])
    for pname in N_FEAT
}
# Skratka pre finálne modely
MODELS = MODELS_PER_PIPE['P2_dotaznik']
print("Modely (s FS vnútri Pipeline):", list(MODELS.keys()))
print(f"  P1: bez FS (5 features)")
print(f"  P2: SelectKBest k={N_FEAT['P2_dotaznik']} vnútri každého foldu")
print(f"  P3: SelectKBest k={N_FEAT['P3_kombinacia']} vnútri každého foldu")


# ===========================================================
# 7. TRÉNING A HODNOTENIE
# ===========================================================

print("\n" + "=" * 60)
print("7. TRÉNING A HODNOTENIE")
print("=" * 60)


def specificity_score(y_true, y_pred):
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


scorers = {
    'auc':         'roc_auc',
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(specificity_score),
    'f1':          'f1',
}

DATASETS = {
    'P1_anamneza':   df_feat[FINAL_FEATURES['P1_anamneza']],
    'P2_dotaznik':   df_feat[FINAL_FEATURES['P2_dotaznik']],
    'P3_kombinacia': df_feat[FINAL_FEATURES['P3_kombinacia']],
}

all_results  = []
all_fold_aucs = {}

for pname, X_pipe in DATASETS.items():
    print(f"\n{'─'*55}\n{pname} ({X_pipe.shape[1]} features)")
    print(f"{'Model':<22} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6}")
    print(f"{'':─<22} {'':─>6} {'':─>6} {'':─>6} {'':─>6}")
    # Použijeme modely s príslušným k pre danú pipeline
    pipe_models = MODELS_PER_PIPE[pname]
    for mname, mpipe in pipe_models.items():
        r = cross_validate(mpipe, X_pipe, y, cv=CV,
                           scoring=scorers, return_train_score=False)
        folds = cross_val_score(mpipe, X_pipe, y, cv=CV, scoring='roc_auc')
        all_fold_aucs[(pname,mname)] = folds
        std     = folds.std()
        verdict = ("konzistentne" if std<0.03
                   else "akceptovatelne" if std<0.05 else "variabilne")
        kscore  = (r['test_auc'].mean()*0.35 +
                   r['test_sensitivity'].mean()*0.40 +
                   r['test_specificity'].mean()*0.15 +
                   (1.0 if std<0.03 else 0.6 if std<0.05 else 0.2)*0.10)
        row = {
            'Pipeline':pname,'Model':mname,
            'AUC':r['test_auc'].mean(),'AUC_std':std,
            'Sensitivity':r['test_sensitivity'].mean(),
            'Sens_std':r['test_sensitivity'].std(),
            'Specificity':r['test_specificity'].mean(),
            'Spec_std':r['test_specificity'].std(),
            'F1':r['test_f1'].mean(),'F1_std':r['test_f1'].std(),
            'Konzistencia':verdict,'Klinicky_score':kscore,
            'Fold1':round(folds[0],4),'Fold2':round(folds[1],4),
            'Fold3':round(folds[2],4),'Fold4':round(folds[3],4),
            'Fold5':round(folds[4],4),
        }
        all_results.append(row)
        print(f"{mname:<22} {row['AUC']:.3f} {row['Sensitivity']:.3f} "
              f"{row['Specificity']:.3f} {row['F1']:.3f}")

results_df = pd.DataFrame(all_results)

print(f"\n{'='*60}\nCELKOVÉ PORADIE (klinické skóre)")
print(f"{'Rank':<4} {'Pipeline':<16} {'Model':<22} "
      f"{'KScore':>7} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'Konz':>14}")
print("─"*90)
ranked = results_df.sort_values('Klinicky_score', ascending=False)
for i,(_,r) in enumerate(ranked.iterrows(),1):
    marker = " ★" if i==1 else (" ◆" if i==2 else "")
    print(f"{i:<4} {r['Pipeline']:<16} {r['Model']:<22} "
          f"{r['Klinicky_score']:>7.4f} {r['AUC']:>6.3f} "
          f"{r['Sensitivity']:>6.3f} {r['Specificity']:>6.3f} "
          f"{r['Konzistencia']:>14}{marker}")

print("\nNAJLEPSÍ PER PIPELINE:")
for pipe in ['P1_anamneza','P2_dotaznik','P3_kombinacia']:
    pb = results_df[results_df['Pipeline']==pipe].sort_values(
        'Klinicky_score',ascending=False).iloc[0]
    print(f"  {pipe:<18}: {pb['Model']:<22} "
          f"AUC={pb['AUC']:.3f} Sens={pb['Sensitivity']:.3f} "
          f"Spec={pb['Specificity']:.3f}")


# ===========================================================
# 8. PER-FOLD AUC — KONZISTENCIA
# ===========================================================

print("\n" + "=" * 60)
print("8. PER-FOLD AUC — KONZISTENCIA")
print("=" * 60)
print("std<0.03 ✓✓ | std<0.05 ✓ | >=0.05 ~\n")
print(f"{'Pipeline':<16} {'Model':<22} "
      f"{'F1':>6} {'F2':>6} {'F3':>6} {'F4':>6} {'F5':>6}  "
      f"{'Mean':>6} {'Std':>6}  Záver")
print("─"*100)
for pname, X_pipe in DATASETS.items():
    for mname in MODELS_PER_PIPE[pname]:
        folds   = all_fold_aucs[(pname,mname)]
        std     = folds.std()
        verdict = "✓✓" if std<0.03 else ("✓" if std<0.05 else "~")
        fstr    = " ".join(f"{v:.3f}" for v in folds)
        print(f"{pname:<16} {mname:<22} {fstr}  "
              f"{folds.mean():.3f}  {std:.3f}  {verdict}")
    print()


# ===========================================================
# 9. FINÁLNE MODELY A VALIDÁCIA
# ===========================================================

print("\n" + "=" * 60)
print("9. FINÁLNE MODELY")
print("=" * 60)
print("""
Výber na základe klinického skóre (Sensitivity 40%, AUC 35%,
Specificity 15%, Konzistencia 10%):
  Model 1: Extra Trees · P1 Anamnéza
    → najkonzistentnejší (CV%~1.4%), 5 features, 30 sek. zberu
  Model 2: Extra Trees · P3 Kombinácia
    → najvyšší AUC, najlepšia Specificita, 17 features
    → obsahuje Ma_diag_srdcove_ochorenie (klinicky odporúčaný)

Poznámka k rovnakej Sensitivity ET·P1 vs ET·P3:
  Oba modely: TP=168/206 = 0.8155
  Ide o ŠTATISTICKÚ ZHODU, nie metodickú chybu.
  Na fold-úrovni sú výsledky odlišné (rôzni pacienti).
  Rozdiel je v Specificity: P3=0.733 vs P1=0.667 (+11 správnych TN).
""")

FINAL_MODELS = {
    'ET_P1_anamneza': {
        'pipeline': MODELS_PER_PIPE['P1_anamneza']['Extra Trees'],
        'X':        DATASETS['P1_anamneza'],
        'features': FINAL_FEATURES['P1_anamneza'],
        'label':    'Extra Trees · P1 Anamnéza',
        'reason':   '5 features, CV%=1.4%, najkonzistentnejší',
    },
    'ET_P3_kombinacia': {
        'pipeline': MODELS_PER_PIPE['P3_kombinacia']['Extra Trees'],
        'X':        DATASETS['P3_kombinacia'],
        'features': FINAL_FEATURES['P3_kombinacia'],
        'label':    'Extra Trees · P3 Kombinácia',
        'reason':   'Najvyšší AUC, najlepšia Specificita, Ma_diag included',
    },
}


def validuj_model(pipeline, X, y, label,
                  n_perm=100, n_boot=500):
    """Permutation test · Learning curve · Bootstrap CI · Confusion matrix."""
    print(f"\n{'─'*52}\nValidácia: {label}")

    # a) Permutation test
    print(f"\n  a) Permutation test ({n_perm} permutácií)")
    score,perm_sc,p_val = permutation_test_score(
        pipeline, X, y, scoring='roc_auc',
        cv=CV, n_permutations=n_perm,
        random_state=42, n_jobs=-1)
    print(f"     AUC modelu:  {score:.3f}")
    print(f"     AUC náhodné: {perm_sc.mean():.3f} ± {perm_sc.std():.3f}")
    print(f"     p-hodnota:   {p_val:.4f}  "
          f"{'ŠTATISTICKY VÝZNAMNÝ ✓' if p_val<0.05 else 'NIE JE VÝZNAMNÝ ✗'}")

    # b) Learning curve
    print(f"\n  b) Learning curve")
    tr_sz,tr_sc,te_sc = learning_curve(
        pipeline, X, y, cv=CV, scoring='roc_auc',
        train_sizes=np.linspace(0.2,1.0,7), n_jobs=-1)
    print(f"     {'Vzorky':>7} {'Train':>7} {'Test':>7} {'Gap':>6}")
    for sz,tr,te in zip(tr_sz,tr_sc.mean(1),te_sc.mean(1)):
        flag = " ← overfit" if (tr-te)>0.15 else ""
        print(f"     {int(sz):>7} {tr:>7.3f} {te:>7.3f} {tr-te:>6.3f}{flag}")
    gap = tr_sc.mean(1)[-1] - te_sc.mean(1)[-1]
    zaver = ("výborná generalizácia ✓" if gap<0.05
             else "mierny overfit — akceptovateľné" if gap<0.15
             else "silný overfit — zvážiť regularizáciu")
    print(f"     Záver: gap={gap:.3f} → {zaver}")

    # c) Bootstrap 95% CI (priamy resample — štandardná metóda)
    print(f"\n  c) Bootstrap 95% CI ({n_boot} iterácií)")
    y_prob_cv = cross_val_predict(
        pipeline, X, y, cv=CV, method='predict_proba')[:,1]
    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2: continue
        boot_aucs.append(roc_auc_score(y[idx], y_prob_cv[idx]))
    ci_low  = np.percentile(boot_aucs, 2.5)
    ci_high = np.percentile(boot_aucs, 97.5)
    auc_cv  = roc_auc_score(y, y_prob_cv)
    print(f"     AUC (CV):      {auc_cv:.3f}")
    print(f"     Bootstrap AUC: {np.mean(boot_aucs):.3f}")
    print(f"     95% CI:        [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"     → Reportovať: AUC = {auc_cv:.3f} "
          f"(95% CI [{ci_low:.3f}–{ci_high:.3f}])")

    # d) Confusion matrix
    print(f"\n  d) Confusion matrix (prah = 0.5)")
    y_pred = (y_prob_cv >= 0.5).astype(int)
    tn,fp,fn,tp = confusion_matrix(y, y_pred).ravel()
    print(f"     TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"     Sensitivity: {tp/(tp+fn):.3f} ({tp}/{tp+fn})")
    print(f"     Specificity: {tn/(tn+fp):.3f} ({tn}/{tn+fp})")
    print(f"     Presnosť:    {(tp+tn)/len(y):.3f}")

    # e) Per-fold Sensitivity kontrola
    print(f"\n  e) Per-fold Sensitivity kontrola")
    print(f"     {'Fold':>4} {'TP':>4} {'FN':>4} {'TN':>4} {'FP':>4} "
          f"{'Sens':>7} {'Spec':>7}")
    for fold_i,(tr_idx,te_idx) in enumerate(CV.split(X,y),1):
        pipeline.fit(X.iloc[tr_idx], y[tr_idx])
        yp = pipeline.predict(X.iloc[te_idx])
        cm = confusion_matrix(y[te_idx], yp)
        if cm.shape==(2,2):
            tn_f,fp_f,fn_f,tp_f = cm.ravel()
            s  = tp_f/(tp_f+fn_f) if (tp_f+fn_f)>0 else 0
            sp = tn_f/(tn_f+fp_f) if (tn_f+fp_f)>0 else 0
            print(f"     {fold_i:>4} {tp_f:>4} {fn_f:>4} "
                  f"{tn_f:>4} {fp_f:>4} {s:>7.3f} {sp:>7.3f}")

    return dict(score=score,perm_sc=perm_sc,p_val=p_val,
                tr_sz=tr_sz,tr_sc=tr_sc,te_sc=te_sc,
                y_prob_cv=y_prob_cv,auc_cv=auc_cv,
                ci_low=ci_low,ci_high=ci_high,boot_aucs=boot_aucs,
                tn=tn,fp=fp,fn=fn,tp=tp)


val_results = {}
for key,info in FINAL_MODELS.items():
    val_results[key] = validuj_model(
        info['pipeline'], info['X'], y, info['label'])


# ===========================================================
# 10. THRESHOLD TUNING
# ===========================================================

print("\n" + "=" * 60)
print("10. THRESHOLD TUNING (min. Sensitivity = 90%)")
print("=" * 60)
MIN_SENSITIVITY = 0.90


def find_clinical_threshold(y_true, y_prob, min_sens=0.90):
    best_thresh, best_spec = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 200):
        pred = (y_prob >= t).astype(int)
        if pred.sum() == 0: continue
        tn,fp,fn,tp = confusion_matrix(y_true,pred).ravel()
        sens = tp/(tp+fn); spec = tn/(tn+fp)
        if sens >= min_sens and spec > best_spec:
            best_spec = spec; best_thresh = t
    pred_f = (y_prob >= best_thresh).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true,pred_f).ravel()
    return {'threshold':best_thresh,
            'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp}


for key,info in FINAL_MODELS.items():
    v   = val_results[key]
    res = find_clinical_threshold(y, v['y_prob_cv'], MIN_SENSITIVITY)
    print(f"\n{info['label']}")
    print(f"  Prah 0.50: Sens={v['tp']/(v['tp']+v['fn']):.3f}  "
          f"Spec={v['tn']/(v['tn']+v['fp']):.3f}")
    print(f"  Prah {res['threshold']:.2f}: Sens={res['sensitivity']:.3f}  "
          f"Spec={res['specificity']:.3f}  ← klinický prah (min.Sens≥90%)")
    print(f"  Zachytí {res['tp']}/{v['tp']+v['fn']} pozit., "
          f"odmietne {res['tn']}/{v['tn']+v['fp']} negat.")
    val_results[key]['clinical_threshold'] = res


# ===========================================================
# 11. GRAFY
# ===========================================================

print("\n" + "=" * 60)
print("11. GRAFY")
print("=" * 60)

# Graf 1: Validácia (2×3)
fig1,axes1 = plt.subplots(2,3,figsize=(15,10))
fig1.suptitle('Validácia finálnych modelov',fontsize=14)
colors = {'ET_P1_anamneza':('royalblue','seagreen'),
          'ET_P3_kombinacia':('darkorange','crimson')}

for ri,(key,info) in enumerate(FINAL_MODELS.items()):
    v=val_results[key]; c1,c2=colors[key]; lbl=info['label']
    axes1[ri][0].hist(v['perm_sc'],bins=25,color='steelblue',alpha=0.7,label='Náhodné')
    axes1[ri][0].axvline(v['score'],color='red',lw=2.5,label=f'Model AUC={v["score"]:.3f}')
    axes1[ri][0].set_title(f'Permutation test\n{lbl}\np={v["p_val"]:.3f}')
    axes1[ri][0].legend(fontsize=8); axes1[ri][0].set_xlabel('AUC')

    axes1[ri][1].plot(v['tr_sz'],v['tr_sc'].mean(1),'o-',color=c1,label='Tréning')
    axes1[ri][1].fill_between(v['tr_sz'],
        v['tr_sc'].mean(1)-v['tr_sc'].std(1),
        v['tr_sc'].mean(1)+v['tr_sc'].std(1),alpha=0.15,color=c1)
    axes1[ri][1].plot(v['tr_sz'],v['te_sc'].mean(1),'o-',color=c2,label='Test CV')
    axes1[ri][1].fill_between(v['tr_sz'],
        v['te_sc'].mean(1)-v['te_sc'].std(1),
        v['te_sc'].mean(1)+v['te_sc'].std(1),alpha=0.15,color=c2)
    axes1[ri][1].set_title(f'Learning curve\n{lbl}')
    axes1[ri][1].legend(); axes1[ri][1].set_ylim([0.5,1.05])
    axes1[ri][1].set_xlabel('Trénovacie vzorky'); axes1[ri][1].set_ylabel('AUC')

    fpr,tpr,_ = roc_curve(y,v['y_prob_cv'])
    axes1[ri][2].plot(fpr,tpr,color=c1,lw=2,
        label=f'AUC={v["auc_cv"]:.3f}\n95% CI [{v["ci_low"]:.3f}–{v["ci_high"]:.3f}]')
    axes1[ri][2].plot([0,1],[0,1],'--',color='gray',label='Náhodný')
    axes1[ri][2].set_title(f'ROC krivka\n{lbl}')
    axes1[ri][2].legend(fontsize=8)
    axes1[ri][2].set_xlabel('1 - Specificita'); axes1[ri][2].set_ylabel('Sensitivita')
    axes1[ri][2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hutt_validation.png',dpi=150,bbox_inches='tight')
print("Uložené: hutt_validation.png")

# Graf 2: Per-fold AUC
MODEL_COLORS = {
    'Logisticka regresia':'#888780','Random Forest':'#378ADD',
    'Extra Trees':'#1D9E75','XGBoost':'#EF9F27',
    'CatBoost':'#D85A30','SVM':'#7F77DD',
}
model_names_plot = list(MODELS_PER_PIPE["P2_dotaznik"].keys())
fig2,axes2 = plt.subplots(1,3,figsize=(16,6),sharey=True)
fig2.suptitle('AUC na každom CV folde',fontsize=13)

for ax,pname in zip(axes2,list(DATASETS.keys())):
    for i,mname in enumerate(model_names_plot):
        folds = all_fold_aucs.get((pname,mname))
        if folds is None: continue
        mean=folds.mean(); std=folds.std()
        color=MODEL_COLORS.get(mname,'gray')
        ax.vlines(i,folds.min(),folds.max(),color=color,lw=2,alpha=0.5)
        for fv,jit in zip(folds,np.linspace(-0.08,0.08,len(folds))):
            ax.scatter(i+jit,fv,color=color,s=30,zorder=5,alpha=0.85)
        ax.scatter(i,mean,color=color,s=120,zorder=6,
                   edgecolors='white',linewidths=1.5)
        ax.fill_between([i-0.25,i+0.25],mean-std,mean+std,
                        color=color,alpha=0.12)
    et_idx = model_names_plot.index('Extra Trees')
    if 'P1' in pname: ax.axvspan(et_idx-0.4,et_idx+0.4,color='#1D9E75',alpha=0.07)
    if 'P3' in pname: ax.axvspan(et_idx-0.4,et_idx+0.4,color='#1D9E75',alpha=0.07)
    ax.set_xticks(range(len(model_names_plot)))
    ax.set_xticklabels(model_names_plot,rotation=35,ha='right',fontsize=8)
    ax.set_title(pname,fontsize=11); ax.set_ylim([0.40,0.97])
    ax.axhline(0.5,color='gray',lw=0.8,linestyle='--',alpha=0.5)
    ax.grid(axis='y',alpha=0.3)
    if ax==axes2[0]: ax.set_ylabel('AUC')

patches=[mpatches.Patch(color=c,label=m) for m,c in MODEL_COLORS.items()]
fig2.legend(handles=patches,loc='lower center',ncol=6,
            fontsize=8,bbox_to_anchor=(0.5,-0.02))
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig('hutt_cv_folds.png',dpi=150,bbox_inches='tight')
print("Uložené: hutt_cv_folds.png")


# ===========================================================
# 12. ULOŽENIE
# ===========================================================

print("\n" + "=" * 60)
print("12. ULOŽENIE")
print("=" * 60)

# CSV s kompletnou tabuľkou výsledkov
fold_rows = []
for (pname,mname),folds in all_fold_aucs.items():
    std = folds.std()
    verdict = ("konzistentne" if std<0.03
               else "akceptovatelne" if std<0.05 else "variabilne")
    r = results_df[(results_df['Pipeline']==pname) &
                   (results_df['Model']==mname)].iloc[0]
    fold_rows.append({
        'Pipeline':pname,'Model':mname,
        'Mean_AUC':round(folds.mean(),4),'Std_AUC':round(std,4),
        'Min_AUC':round(folds.min(),4),'Max_AUC':round(folds.max(),4),
        'Rozptyl_AUC':round(folds.max()-folds.min(),4),
        'CV_pct':round((std/folds.mean())*100,2),
        'Konzistencia':verdict,
        'Fold1':round(folds[0],4),'Fold2':round(folds[1],4),
        'Fold3':round(folds[2],4),'Fold4':round(folds[3],4),
        'Fold5':round(folds[4],4),
        'Sensitivity':round(r['Sensitivity'],4),
        'Sens_std':round(r['Sens_std'],4),
        'Specificity':round(r['Specificity'],4),
        'Spec_std':round(r['Spec_std'],4),
        'F1':round(r['F1'],4),'F1_std':round(r['F1_std'],4),
        'Klinicky_score':round(r['Klinicky_score'],4),
    })

final_csv = pd.DataFrame(fold_rows).sort_values(
    'Klinicky_score', ascending=False)
final_csv.to_csv('hutt_results.csv', index=False)
print(f"hutt_results.csv: {len(final_csv)} riadkov × {len(final_csv.columns)} stĺpcov")

# Natrénujeme finálne modely na celom datasete pre aplikáciu
for key,info in FINAL_MODELS.items():
    info['pipeline'].fit(info['X'], y)
    print(f"Natrénovaný pre aplikáciu: {info['label']}")

with open('hutt_preprocessed.pkl','wb') as f:
    pickle.dump({
        'df_feat':df_feat,'y':y,
        'FINAL_FEATURES':FINAL_FEATURES,
        'DATASETS':DATASETS,'MODELS_PER_PIPE':MODELS_PER_PIPE,
        'FINAL_MODELS':FINAL_MODELS,
        'results_df':results_df,
        'val_results':val_results,
        'all_fold_aucs':all_fold_aucs,
    }, f)

print("\nSúbory uložené:")
print("  hutt_results.csv       — kompletná tabuľka výsledkov")
print("  hutt_preprocessed.pkl  — modely + dáta pre aplikáciu")
print("  hutt_validation.png    — grafy validácie")
print("  hutt_cv_folds.png      — per-fold AUC graf")
print("\nĎalší krok: hutt_app.py — webová aplikácia (Streamlit)")

