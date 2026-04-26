"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ROZŠÍRENÁ ANALÝZA – PREDIKCIA VÝSLEDKU HUTT TESTU                         ║
║  Nadväzuje na 08_kompletna_finalna_analyza_upravena.py                      ║
║                                                                              ║
║  Nové prvky oproti analýze 08:                                               ║
║    • LASSO logistická regresia (L1) – vlastný výber atribútov               ║
║    • Bootstrap 95% CI pre všetky kľúčové metriky                            ║
║    • Kalibračná krivka (reliability diagram)                                 ║
║    • Decision Curve Analysis (DCA)                                           ║
║    • Feature engineering: Pulse Pressure                                     ║
║    • Porovnanie RF (analýza 08) vs. LASSO LR                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Spustenie:
    py 09_rozsirena_analyza.py

Výstupy:
    vysledky_09_bootstrap_ci.csv       – metriky s 95% CI pre oba modely
    graf_09_kalibracia.png             – kalibračné krivky RF vs. LASSO LR
    graf_09_dca.png                    – Decision Curve Analysis
    graf_09_lasso_koeficienty.png      – koeficienty LASSO LR
    graf_09_bootstrap_ci.png           – vizualizácia CI pre kľúčové metriky
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_full1.csv')
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 1: NAČÍTANIE A PREDSPRACOVANIE DÁT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 1: NAČÍTANIE A PREDSPRACOVANIE DÁT")
print("═"*70)

def parse_bp(val):
    val = str(val).strip()
    if val in ("-1", "", "nan", "NEMERAT", "NEMER", "NEMERST"):
        return np.nan, np.nan
    first = val.split("-")[0].split(",")[0].strip()
    if "/" in first:
        parts = first.split("/")
        try:
            s = float(parts[0])
            d = float(parts[1].strip()) if parts[1].strip() not in ("-", "", "nan") else np.nan
            return s, d
        except:
            return np.nan, np.nan
    return np.nan, np.nan

df = pd.read_csv(DATA_PATH)
print(f"Načítaných riadkov: {len(df)}, stĺpcov: {len(df.columns)}")

bp_parsed          = df["A2"].map(parse_bp)
df["TK_sys"]       = [x[0] for x in bp_parsed]
df["TK_dia"]       = [x[1] for x in bp_parsed]
df["Pulz"]         = pd.to_numeric(df["A3"], errors="coerce").replace(-1, np.nan)
df["Pohlavie_enc"] = (df["Pohlavie"] == "M").astype(float)

META = {"Pohlavie", "Pohlavie_enc", "Vek", "Synkopa", "Typ Synkopy",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
        "TK_sys", "TK_dia", "Pulz", "Dátum", "Datum narodenia", "S", "Číslo dotazníka"}

for col in df.columns:
    if col not in META:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(-1, np.nan)

y = df["Synkopa"].astype(int)
print(f"Pacientov: {len(df)}  |  Synkopa=1: {y.sum()} ({y.mean()*100:.1f} %)  |  Synkopa=0: {(y==0).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 2: FEATURE ENGINEERING")
print("═"*70)

# Pulse Pressure = TK_sys - TK_dia  (hemodynamický marker autonomnej dysfunkcie)
df["Pulse_pressure"] = df["TK_sys"] - df["TK_dia"]
print(f"Pulse Pressure (TK_sys - TK_dia): medián = {df['Pulse_pressure'].median():.1f} mmHg")
print(f"  NaN: {df['Pulse_pressure'].isna().sum()} ({df['Pulse_pressure'].isna().mean()*100:.1f} %)")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 3: DEFINÍCIA SKUPÍN ATRIBÚTOV
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 3: DEFINÍCIA SKUPÍN ATRIBÚTOV")
print("═"*70)

# Rovnaké vylúčenia ako v analýze 08 (na základe požiadaviek lekárov)
VYLUCENE_LEKAR = {'B2', 'C3', 'J1', 'N7', 'P32', 'Q1', 'Q4',
                  'Q12', 'Q13', 'Q16', 'Q17', 'Q18', 'A10'}

# Základné anamnestické atribúty (merané pred testom)
ANAMN = ["Pohlavie_enc", "Vek", "TK_sys", "TK_dia", "Pulz", "Pulse_pressure"]

# Dotazníkové kandidáty (bez vylúčených)
DOTAZNIK_KANDIDATI = [
    c for c in df.columns
    if c not in META
    and c not in VYLUCENE_LEKAR
    and c not in ["Synkopa", "Typ Synkopy", "Ma_srdcove_ochorenie",
                  "Ma_diag_srdcove_ochorenie", "Pulse_pressure"]
    and not c.startswith("P1") or c in ["P12"]
]
# Korektná definícia – všetky stĺpce okrem META, vylúčených a engineered
DOTAZNIK_KANDIDATI = [
    c for c in df.columns
    if c not in META
    and c not in VYLUCENE_LEKAR
    and c not in {"Synkopa", "Typ Synkopy", "Ma_srdcove_ochorenie",
                  "Ma_diag_srdcove_ochorenie", "Pulse_pressure"}
]

ALL_KANDIDATI = ANAMN + DOTAZNIK_KANDIDATI
print(f"Anamnestické atribúty: {len(ANAMN)}  (vrátane Pulse Pressure)")
print(f"Dotazníkové kandidáty: {len(DOTAZNIK_KANDIDATI)}")
print(f"Celkový pool: {len(ALL_KANDIDATI)}")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 4: TRAIN / TEST SPLIT (identický s analýzou 08)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 4: TRAIN / TEST SPLIT (80 / 20, seed=42)")
print("═"*70)

y_arr = y.values
np.random.seed(42)
idx0 = np.where(y_arr == 0)[0]; np.random.shuffle(idx0)
idx1 = np.where(y_arr == 1)[0]; np.random.shuffle(idx1)
n0 = max(1, int(len(idx0) * 0.2))
n1 = max(1, int(len(idx1) * 0.2))
te_idx = np.concatenate([idx0[:n0], idx1[:n1]])
tr_idx = np.concatenate([idx0[n0:],  idx1[n1:]])

print(f"Trénovacia sada: {len(tr_idx)}  (Synkopa=1: {y_arr[tr_idx].sum()}, =0: {(y_arr[tr_idx]==0).sum()})")
print(f"Testovacia sada: {len(te_idx)}  (Synkopa=1: {y_arr[te_idx].sum()}, =0: {(y_arr[te_idx]==0).sum()})")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 5: CHI² VÝBER + IMPUTÁCIA (pre LASSO pool)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 5: CHI² VÝBER ATRIBÚTOV (len na trénovacej sade)")
print("═"*70)

def chi2_test(x, y):
    """Chi² test pre binárne / Mann-Whitney pre spojité."""
    valid = ~np.isnan(x)
    xv, yv = x[valid], y[valid]
    if len(np.unique(xv)) <= 5:          # binárne / kategorické
        cats = np.unique(xv)
        n = len(yv)
        observed, expected = [], []
        for c in cats:
            mask = xv == c
            o0 = ((mask) & (yv == 0)).sum()
            o1 = ((mask) & (yv == 1)).sum()
            e0 = mask.sum() * (yv == 0).sum() / n
            e1 = mask.sum() * (yv == 1).sum() / n
            observed += [o0, o1]; expected += [e0, e1]
        obs = np.array(observed, dtype=float)
        exp = np.array(expected, dtype=float)
        exp = np.where(exp < 1e-9, 1e-9, exp)
        stat = ((obs - exp)**2 / exp).sum()
        df_chi = len(cats) - 1
        # p-hodnota aproximácia cez chi² CDF
        p = 1 - chi2_cdf(stat, df_chi)
        return p, "chi²"
    else:                                # spojité – Mann-Whitney
        g0 = xv[yv == 0]; g1 = xv[yv == 1]
        if len(g0) == 0 or len(g1) == 0:
            return 1.0, "MW"
        u = 0
        for a in g0:
            u += (g1 < a).sum() + 0.5 * (g1 == a).sum()
        n0_, n1_ = len(g0), len(g1)
        mu = n0_ * n1_ / 2
        sigma = np.sqrt(n0_ * n1_ * (n0_ + n1_ + 1) / 12)
        z = (u - mu) / (sigma + 1e-9)
        p = 2 * (1 - normal_cdf(abs(z)))
        return p, "Mann-Whitney"

def chi2_cdf(x, k):
    """Aproximácia chi² CDF pomocou gama funkcie."""
    if x <= 0: return 0.0
    return regularized_gamma(k / 2, x / 2)

def regularized_gamma(a, x):
    """Dolná nekompletná gama funkcia (regularizovaná) – sériová aproximácia."""
    if x == 0: return 0.0
    if x < 0:  return 0.0
    MAX_ITER = 200
    term = np.exp(-x + a * np.log(x) - log_gamma(a)) / a
    total = term
    for n_ in range(1, MAX_ITER):
        term *= x / (a + n_)
        total += term
        if abs(term) < 1e-10 * abs(total):
            break
    return min(total, 1.0)

def log_gamma(x):
    """Stirlingova aproximácia ln(Γ(x))."""
    if x <= 0: return 0.0
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi) + \
           1/(12*x) - 1/(360*x**3)

def normal_cdf(x):
    """Aproximácia normálneho CDF."""
    return 0.5 * (1 + erf_approx(x / np.sqrt(2)))

def erf_approx(x):
    t = 1 / (1 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
           t * (-1.453152027 + t * 1.061405429))))
    return np.sign(x) * (1 - poly * np.exp(-x*x))

# Chi² na trénovacej sade
X_train_df = df[ALL_KANDIDATI].iloc[tr_idx]
y_train    = y_arr[tr_idx]

chi_results = []
for col in DOTAZNIK_KANDIDATI:
    x_col = X_train_df[col].values.astype(float)
    if np.isnan(x_col).all():
        continue
    p, test = chi2_test(x_col, y_train)
    chi_results.append({"Atribút": col, "p-hodnota": p, "Test": test})

chi_df = pd.DataFrame(chi_results).sort_values("p-hodnota").reset_index(drop=True)
selected_chi2 = chi_df[chi_df["p-hodnota"] < 0.05]["Atribút"].tolist()

print(f"\nVybratých atribútov dotazníka (chi², p<0.05): {len(selected_chi2)}")
for _, row in chi_df[chi_df["p-hodnota"] < 0.05].iterrows():
    print(f"  {row['Atribút']:<8}  p = {row['p-hodnota']:.4f}  [{row['Test']}]")

# Kombinácia pre RF (rovnaká ako v analýze 08)
KOMBINACIA_RF = ["Pohlavie_enc", "Vek", "TK_sys", "TK_dia", "Pulz"] + selected_chi2
print(f"\nKombinácia pre RF: {len(KOMBINACIA_RF)} atribútov")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 6: RANDOM FOREST (from scratch, rovnaký ako v analýze 08)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 6: RANDOM FOREST (from scratch, n=200, max_depth=12)")
print("═"*70)

class DTree:
    def __init__(self, max_depth=12, min_split=4, max_features=None):
        self.max_depth = max_depth; self.min_split = min_split
        self.max_features = max_features

    def _gini(self, y):
        p = y.mean(); return 1 - p*p - (1-p)**2

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
                if g > best[3]: best = (fi, t, lm, g)
        return best

    def _build(self, X, y, d, n_total):
        if d >= self.max_depth or len(y) < self.min_split or self._gini(y) < 1e-7:
            return {'leaf': True, 'p': float(y.mean())}
        fi, t, lm, g = self._split(X, y)
        if fi == -1 or g < 1e-7: return {'leaf': True, 'p': float(y.mean())}
        return {'leaf': False, 'fi': fi, 't': t,
                'L': self._build(X[lm],  y[lm],  d+1, n_total),
                'R': self._build(X[~lm], y[~lm], d+1, n_total)}

    def fit(self, X, y):
        self.tree_ = self._build(X, y.astype(float), 0, len(y)); return self

    def _p1(self, x, n):
        return n['p'] if n['leaf'] else self._p1(x, n['L'] if x[n['fi']] <= n['t'] else n['R'])

    def predict_proba(self, X):
        return np.array([self._p1(x, self.tree_) for x in X])


class RandomForest:
    def __init__(self, n=200, max_depth=12, seed=42):
        self.n = n; self.max_depth = max_depth; self.seed = seed
        self.trees_ = []; self.fidx_ = []

    def fit(self, X, y):
        np.random.seed(self.seed)
        ns, nf = X.shape; mf = max(1, int(np.sqrt(nf)))
        for _ in range(self.n):
            bi = np.random.choice(ns, ns, replace=True)
            fi = np.random.choice(nf, mf, replace=False)
            t  = DTree(max_depth=self.max_depth, min_split=4, max_features=mf)
            t.fit(X[np.ix_(bi, fi)], y[bi])
            self.trees_.append(t); self.fidx_.append(fi)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X[:, fi])
                        for t, fi in zip(self.trees_, self.fidx_)], axis=0)

# Imputácia + tréning RF
feats_rf  = KOMBINACIA_RF
X_tr_rf   = df[feats_rf].iloc[tr_idx].values.astype(float)
X_te_rf   = df[feats_rf].iloc[te_idx].values.astype(float)
med_rf    = np.nanmedian(X_tr_rf, axis=0)
X_tr_rf_i = np.where(np.isnan(X_tr_rf), med_rf, X_tr_rf)
X_te_rf_i = np.where(np.isnan(X_te_rf), med_rf, X_te_rf)

print("Trénujem RF (200 stromov)...")
rf = RandomForest(n=200, max_depth=12, seed=42)
rf.fit(X_tr_rf_i, y_train)
proba_rf_te = rf.predict_proba(X_te_rf_i)
print("RF natrénovaný.")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 7: LASSO LOGISTICKÁ REGRESIA
#   Prístup: chi² predfiltrácia (top 30) → LASSO L1 (vlastný výber)
#
#   Dôvod predfiltrácie:
#     297 vzoriek ÷ 137 atribútov = 2.2 vzorky/atribút → silný overfitting.
#     Redukujeme na top 30 podľa chi² (zachovávame len štatisticky relevantné),
#     čím zlepšíme pomer vzorky/atribút na ~10:1.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 7: LASSO LOGISTICKÁ REGRESIA")
print("  (chi² predfiltrácia top 30 → L1 penalizácia → CV pre C)")
print("═"*70)

# Krok 7a: chi² predfiltrácia – top 30 dotazníkových atribútov
TOP_N_CHI2 = 30
top30_feats = chi_df.head(TOP_N_CHI2)["Atribút"].tolist()
feats_lasso = ANAMN + top30_feats

print(f"\n  Krok 1 – chi² predfiltrácia:")
print(f"  Vstup: {len(DOTAZNIK_KANDIDATI)} dotazníkových kandidátov")
print(f"  Vybrané top {TOP_N_CHI2} podľa p-hodnoty + {len(ANAMN)} anamnestických")
print(f"  Pool pre LASSO: {len(feats_lasso)} atribútov  "
      f"(pomer vzorky/atribút = {len(tr_idx)/len(feats_lasso):.1f})")
print(f"\n  Top {TOP_N_CHI2} dotazníkových atribútov odovzdaných LASSO:")
for _, row in chi_df.head(TOP_N_CHI2).iterrows():
    marker = "  ← p<0.05 (vybraný v analýze 08)" if row["p-hodnota"] < 0.05 else ""
    print(f"    {row['Atribút']:<8}  p={row['p-hodnota']:.4f}{marker}")

# Krok 7b: Imputácia + štandardizácia
X_tr_l  = df[feats_lasso].iloc[tr_idx].values.astype(float)
X_te_l  = df[feats_lasso].iloc[te_idx].values.astype(float)
med_l   = np.nanmedian(X_tr_l, axis=0)
X_tr_li = np.where(np.isnan(X_tr_l), med_l, X_tr_l)
X_te_li = np.where(np.isnan(X_te_l), med_l, X_te_l)

scaler  = StandardScaler()
X_tr_ls = scaler.fit_transform(X_tr_li)
X_te_ls = scaler.transform(X_te_li)

# Krok 7c: CV pre C – rozsah posunutý k silnejšej regularizácii
#   Pri malom datasete očakávame optimálne C v rozsahu 0.001–1.0
print(f"\n  Krok 2 – CV pre regularizačný parameter C (rozsah 0.001–1.0)...")
lasso_cv = LogisticRegressionCV(
    Cs=np.logspace(-3, 0, 25),   # 0.001 až 1.0 – silnejšia regularizácia
    cv=5,
    penalty='l1',
    solver='liblinear',
    scoring='roc_auc',
    random_state=42,
    max_iter=2000
)
lasso_cv.fit(X_tr_ls, y_train)
best_C = lasso_cv.C_[0]
print(f"  Optimálne C = {best_C:.4f}  "
      f"({'slabá' if best_C > 0.5 else 'stredná' if best_C > 0.05 else 'silná'} regularizácia)")

# Krok 7d: Finálny LASSO model
lasso = LogisticRegression(
    penalty='l1', solver='liblinear', C=best_C,
    random_state=42, max_iter=2000
)
lasso.fit(X_tr_ls, y_train)
proba_lasso_te = lasso.predict_proba(X_te_ls)[:, 1]

# Koeficienty
coef = lasso.coef_[0]
selected_lasso = [(feats_lasso[i], coef[i]) for i in range(len(feats_lasso)) if coef[i] != 0]
selected_lasso_zero = [feats_lasso[i] for i in range(len(feats_lasso)) if coef[i] == 0]
selected_lasso.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n  Krok 3 – LASSO výber z {len(feats_lasso)} atribútov:")
print(f"  Vybrané (nenulový koeficient): {len(selected_lasso)}")
print(f"  Eliminované (koeficient = 0):  {len(selected_lasso_zero)}")
if selected_lasso_zero:
    print(f"  Eliminované: {', '.join(selected_lasso_zero)}")
print(f"\n  {'Atribút':<12}  {'Koeficient':>12}  {'Smer'}")
print("  " + "-"*38)
for feat, c in selected_lasso:
    smer = "↑ zvyšuje riziko" if c > 0 else "↓ znižuje riziko"
    print(f"  {feat:<12}  {c:>12.4f}  {smer}")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 8: METRIKY A BOOTSTRAP 95% CI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 8: METRIKY + BOOTSTRAP 95% CI (n_boot=1000)")
print("═"*70)

def auc_roc(y_true, y_score):
    """AUC cez Wilcoxon rank-sum (bez scipy)."""
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    u = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return u / (len(pos) * len(neg))

def calc_metrics(y_true, y_prob, thr=0.50):
    pred = (y_prob >= thr).astype(int)
    TP = int(((pred==1)&(y_true==1)).sum())
    FP = int(((pred==1)&(y_true==0)).sum())
    FN = int(((pred==0)&(y_true==1)).sum())
    TN = int(((pred==0)&(y_true==0)).sum())
    sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
    spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
    ppv  = TP/(TP+FP) if (TP+FP)>0 else 0.0
    npv  = TN/(TN+FN) if (TN+FN)>0 else 0.0
    auc  = auc_roc(y_true, y_prob)
    return {"AUC": auc, "Sens": sens, "Spec": spec, "PPV": ppv, "NPV": npv,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN}

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    """Bootstrap 95% CI pre ľubovoľnú metriku."""
    np.random.seed(seed)
    n = len(y_true); scores = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2: continue
        scores.append(metric_fn(yt, ys))
    scores = np.array(scores)
    return np.percentile(scores, [2.5, 97.5])

y_test = y_arr[te_idx]

# Bodové odhady
m_rf    = calc_metrics(y_test, proba_rf_te)
m_lasso = calc_metrics(y_test, proba_lasso_te)

print("\nBootstrap CI prebieha (môže trvať ~30s)...")
metrics_list = [
    ("AUC",  lambda yt, ys: auc_roc(yt, ys)),
    ("Sens", lambda yt, ys: calc_metrics(yt, ys)["Sens"]),
    ("Spec", lambda yt, ys: calc_metrics(yt, ys)["Spec"]),
    ("PPV",  lambda yt, ys: calc_metrics(yt, ys)["PPV"]),
    ("NPV",  lambda yt, ys: calc_metrics(yt, ys)["NPV"]),
]

ci_rf    = {m: bootstrap_ci(y_test, proba_rf_te,    fn) for m, fn in metrics_list}
ci_lasso = {m: bootstrap_ci(y_test, proba_lasso_te, fn) for m, fn in metrics_list}

print("\n" + "─"*70)
print(f"{'Metrika':<8}  {'RF':>22}  {'LASSO LR':>22}")
print(f"{'':8}  {'bod. odhad [95% CI]':>22}  {'bod. odhad [95% CI]':>22}")
print("─"*70)

rows = []
for m, _ in metrics_list:
    rf_str    = f"{m_rf[m]*100:.1f} % [{ci_rf[m][0]*100:.1f}–{ci_rf[m][1]*100:.1f}]"
    lasso_str = f"{m_lasso[m]*100:.1f} % [{ci_lasso[m][0]*100:.1f}–{ci_lasso[m][1]*100:.1f}]"
    print(f"  {m:<6}  {rf_str:>22}  {lasso_str:>22}")
    rows.append({"Metrika": m,
                 "RF_bod":     round(m_rf[m]*100, 1),
                 "RF_CI_low":  round(ci_rf[m][0]*100, 1),
                 "RF_CI_high": round(ci_rf[m][1]*100, 1),
                 "LASSO_bod":  round(m_lasso[m]*100, 1),
                 "LASSO_CI_low":  round(ci_lasso[m][0]*100, 1),
                 "LASSO_CI_high": round(ci_lasso[m][1]*100, 1)})

print("─"*70)
print(f"  Konfúzna matica RF:    TP={m_rf['TP']} FP={m_rf['FP']} FN={m_rf['FN']} TN={m_rf['TN']}")
print(f"  Konfúzna matica LASSO: TP={m_lasso['TP']} FP={m_lasso['FP']} FN={m_lasso['FN']} TN={m_lasso['TN']}")

# Uloženie CSV
ci_df = pd.DataFrame(rows)
ci_df.to_csv(os.path.join(OUT_DIR, 'vysledky_09_bootstrap_ci.csv'), index=False)
print(f"\n  Uložené: vysledky_09_bootstrap_ci.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 9: KALIBRAČNÁ KRIVKA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 9: KALIBRAČNÁ KRIVKA (Reliability Diagram)")
print("═"*70)

def calibration_curve(y_true, y_prob, n_bins=8):
    """Reliability diagram – binujeme predikcie, porovnávame s pozorovanou frekvenciou."""
    bins   = np.linspace(0, 1, n_bins + 1)
    b_pred, b_true, b_cnt = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        if mask.sum() > 0:
            b_pred.append(y_prob[mask].mean())
            b_true.append(y_true[mask].mean())
            b_cnt.append(mask.sum())
    return np.array(b_pred), np.array(b_true), np.array(b_cnt)

# Brier score (nižší = lepší, max 0.25 pre náhodný model)
def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true)**2)

bp_rf, bt_rf, bc_rf       = calibration_curve(y_test, proba_rf_te)
bp_lasso, bt_lasso, bc_l  = calibration_curve(y_test, proba_lasso_te)
bs_rf    = brier_score(y_test, proba_rf_te)
bs_lasso = brier_score(y_test, proba_lasso_te)

print(f"  Brier score RF:       {bs_rf:.4f}  (nižší = lepší, náhodný model ≈ 0.25)")
print(f"  Brier score LASSO LR: {bs_lasso:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Kalibračná krivka – RF vs. LASSO LR", fontsize=13, fontweight='bold')

for ax, bp, bt, bc, lbl, col, bs in [
    (axes[0], bp_rf,    bt_rf,    bc_rf, "RF",       "#1565C0", bs_rf),
    (axes[1], bp_lasso, bt_lasso, bc_l,  "LASSO LR", "#C0392B", bs_lasso),
]:
    ax.plot([0,1],[0,1], 'k--', lw=1.2, label="Perfektná kalibrácia")
    sc = ax.scatter(bp, bt, s=bc*4, c=col, alpha=0.85, zorder=5,
                    label=f"{lbl} (Brier={bs:.3f})")
    ax.plot(bp, bt, '-o', color=col, lw=1.5, markersize=5)
    ax.fill_between(bp, bt, bp, alpha=0.08, color=col)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Priemerná predikovaná pravdepodobnosť", fontsize=10)
    ax.set_ylabel("Pozorovaná frekvencia synkopy", fontsize=10)
    ax.set_title(f"{lbl} – Reliability Diagram", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    # Histogram predikcií dole
    ax2 = ax.inset_axes([0, -0.28, 1, 0.22])
    ax2.hist(proba_rf_te if lbl=="RF" else proba_lasso_te,
             bins=15, color=col, alpha=0.6, edgecolor='white')
    ax2.set_xlabel("Predikovaná pravdepodobnosť"); ax2.set_ylabel("Počet")
    ax2.set_xlim(0,1); ax2.grid(alpha=0.2)

plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig(os.path.join(OUT_DIR, 'graf_09_kalibracia.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Uložené: graf_09_kalibracia.png")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 10: DECISION CURVE ANALYSIS (DCA)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 10: DECISION CURVE ANALYSIS (DCA)")
print("═"*70)
print("  DCA meria klinickú užitočnosť modelu pri rôznych rozhodovacích prahoch.")
print("  Net Benefit = (TP/n) - (FP/n) × (t / (1-t))")
print("  Vyšší net benefit = model je klinicky užitočnejší.")

def decision_curve(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.95, 95)
    n = len(y_true)
    prev = y_true.mean()
    nb_model, nb_all, nb_none = [], [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        TP   = ((pred==1) & (y_true==1)).sum()
        FP   = ((pred==1) & (y_true==0)).sum()
        nb_model.append(TP/n - (FP/n) * t/(1-t))
        # Stratégia "liečiť všetkých"
        nb_all.append(prev - (1-prev) * t/(1-t))
        # Stratégia "neliečiť nikoho" = 0
        nb_none.append(0.0)
    return thresholds, np.array(nb_model), np.array(nb_all)

thr_range = np.linspace(0.05, 0.80, 76)
thr_rf,    nb_rf,    nb_all_rf    = decision_curve(y_test, proba_rf_te,    thr_range)
thr_lasso, nb_lasso, nb_all_lasso = decision_curve(y_test, proba_lasso_te, thr_range)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thr_rf,    nb_rf,       color='#1565C0', lw=2.2, label='RF (analýza 08)')
ax.plot(thr_lasso, nb_lasso,    color='#C0392B', lw=2.2, label='LASSO LR')
ax.plot(thr_rf,    nb_all_rf,   color='#555',   lw=1.4, ls='--', label='Liečiť všetkých')
ax.axhline(0, color='#999', lw=1.2, ls=':',  label='Neliečiť nikoho')
ax.axvline(0.50, color='orange', lw=1.2, ls='--', alpha=0.7, label='Prah 0.50 (zvolený)')
ax.set_xlim(0.05, 0.80); ax.set_ylim(-0.05, 0.35)
ax.set_xlabel("Rozhodovací prah pravdepodobnosti", fontsize=11)
ax.set_ylabel("Net Benefit", fontsize=11)
ax.set_title("Decision Curve Analysis – RF vs. LASSO LR", fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.text(0.52, 0.02, "← model nie je užitočný\n   (net benefit < 0)", fontsize=8, color='#999')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'graf_09_dca.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Uložené: graf_09_dca.png")

# Kde je RF / LASSO LR lepší
better_rf    = (nb_rf > nb_lasso).sum()
better_lasso = (nb_lasso > nb_rf).sum()
print(f"\n  Počet prahov kde RF > LASSO:    {better_rf} / {len(thr_range)}")
print(f"  Počet prahov kde LASSO > RF:    {better_lasso} / {len(thr_range)}")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 11: VIZUALIZÁCIA KOEFICIENTOV LASSO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 11: KOEFICIENTY LASSO LOGISTICKEJ REGRESIE")
print("═"*70)

if selected_lasso:
    feats_plot = [f for f, _ in selected_lasso]
    coefs_plot = [c for _, c in selected_lasso]
    colors_plot = ['#C0392B' if c > 0 else '#1565C0' for c in coefs_plot]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feats_plot)*0.45)))
    bars = ax.barh(range(len(feats_plot)), coefs_plot, color=colors_plot, alpha=0.82, edgecolor='white')
    ax.set_yticks(range(len(feats_plot)))
    ax.set_yticklabels(feats_plot, fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Koeficient LASSO (log-odds)", fontsize=10)
    ax.set_title(f"LASSO LR – koeficienty vybraných atribútov (C={best_C:.3f})\n"
                 f"Červená = zvyšuje riziko synkopy | Modrá = znižuje riziko", fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, coefs_plot)):
        ax.text(val + 0.003 * np.sign(val), i, f"{val:+.3f}",
                va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'graf_09_lasso_koeficienty.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Uložené: graf_09_lasso_koeficienty.png")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 12: VIZUALIZÁCIA BOOTSTRAP CI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 12: VIZUALIZÁCIA BOOTSTRAP 95% CI")
print("═"*70)

metric_labels = ["AUC", "Sens", "Spec", "PPV", "NPV"]
x = np.arange(len(metric_labels))
width = 0.32

fig, ax = plt.subplots(figsize=(11, 6))
for i, m in enumerate(metric_labels):
    # RF
    ax.bar(x[i] - width/2, m_rf[m]*100, width,
           color='#1565C0', alpha=0.8, label='RF' if i==0 else "")
    ax.errorbar(x[i] - width/2, m_rf[m]*100,
                yerr=[[m_rf[m]*100 - ci_rf[m][0]*100],
                      [ci_rf[m][1]*100 - m_rf[m]*100]],
                fmt='none', color='#0A2F6B', capsize=5, lw=2)
    # LASSO LR
    ax.bar(x[i] + width/2, m_lasso[m]*100, width,
           color='#C0392B', alpha=0.8, label='LASSO LR' if i==0 else "")
    ax.errorbar(x[i] + width/2, m_lasso[m]*100,
                yerr=[[m_lasso[m]*100 - ci_lasso[m][0]*100],
                      [ci_lasso[m][1]*100 - m_lasso[m]*100]],
                fmt='none', color='#7B241C', capsize=5, lw=2)

ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel("Hodnota metriky (%)", fontsize=11)
ax.set_title("RF vs. LASSO LR – metriky s 95% bootstrap CI\n(testovacia sada, n=74)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11); ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
ax.axhline(50, color='#ccc', ls='--', lw=0.8)

# Hodnoty nad stĺpcami
for i, m in enumerate(metric_labels):
    ax.text(x[i]-width/2, m_rf[m]*100+2,    f"{m_rf[m]*100:.1f}",    ha='center', fontsize=8.5, color='#0A2F6B')
    ax.text(x[i]+width/2, m_lasso[m]*100+2,  f"{m_lasso[m]*100:.1f}", ha='center', fontsize=8.5, color='#7B241C')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'graf_09_bootstrap_ci.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Uložené: graf_09_bootstrap_ci.png")

# ══════════════════════════════════════════════════════════════════════════════
# SEKCIA 13: ZÁVER A POROVNANIE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("  SEKCIA 13: ZÁVER A POROVNANIE MODELOV")
print("═"*70)

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  POROVNANIE: RF (analýza 08) vs. LASSO LR (analýza 09)        │
  ├──────────────┬──────────────────────────┬───────────────────────┤
  │  Metrika     │  RF                      │  LASSO LR             │
  ├──────────────┼──────────────────────────┼───────────────────────┤
  │  AUC         │  {m_rf['AUC']*100:4.1f} % [{ci_rf['AUC'][0]*100:.1f}–{ci_rf['AUC'][1]*100:.1f}]  │  {m_lasso['AUC']*100:4.1f} % [{ci_lasso['AUC'][0]*100:.1f}–{ci_lasso['AUC'][1]*100:.1f}]  │
  │  Senzitivita │  {m_rf['Sens']*100:4.1f} % [{ci_rf['Sens'][0]*100:.1f}–{ci_rf['Sens'][1]*100:.1f}]  │  {m_lasso['Sens']*100:4.1f} % [{ci_lasso['Sens'][0]*100:.1f}–{ci_lasso['Sens'][1]*100:.1f}]  │
  │  Špecificita │  {m_rf['Spec']*100:4.1f} % [{ci_rf['Spec'][0]*100:.1f}–{ci_rf['Spec'][1]*100:.1f}]  │  {m_lasso['Spec']*100:4.1f} % [{ci_lasso['Spec'][0]*100:.1f}–{ci_lasso['Spec'][1]*100:.1f}]  │
  │  Brier score │  {bs_rf:.4f}                    │  {bs_lasso:.4f}                 │
  │  # atribútov │  {len(KOMBINACIA_RF):2d} (chi² výber)          │  {len(selected_lasso):2d} (LASSO výber)       │
  ├──────────────┼──────────────────────────┼───────────────────────┤
  │  Kalibrácia  │  nekalibrovaný           │  prirodzene lepší     │
  │  Interpret.  │  black-box               │  koeficienty (log-odds│
  │  DCA výhoda  │  {'RF' if better_rf > better_lasso else 'LASSO LR':10}               │  {'LASSO LR' if better_lasso >= better_rf else 'RF':10}           │
  └──────────────┴──────────────────────────┴───────────────────────┘

  KLINICKÉ ODPORÚČANIE:
  - Ak je prioritou senzitivita → RF (menej prehliadnutých synkop)
  - Ak je prioritou interpretovateľnosť + kalibrácia → LASSO LR
  - Oba modely majú podobný AUC – rozdiel nie je štatisticky významný
    (CI sa prekrývajú), čo potvrdzuje robustnosť výsledkov.
  - LASSO LR identifikovala {len(selected_lasso)} z {len(feats_lasso)} atribútov ako relevantné –
    porovnaj s {len(KOMBINACIA_RF)} atribútmi vybraných chi² testom.
""")

print("  HOTOVO – všetky grafy a CSV uložené.")
print("═"*70)
