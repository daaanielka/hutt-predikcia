"""
Exploratory Data Analysis — hutt_analysis
===============================================
Bakalárska práca: Predikcia výsledku HUTT testu

Obsah:
  1. Načítanie a predspracovanie
  2. Analýza chýbajúcich dát
  3. Opis datasetu — demografia
  4. Distribúcia cieľovej premennej
  5. Numerické features — distribúcie a porovnanie HUTT+/HUTT-
  6. Binárne features — frekvencia a asociácia s A10
  7. Štatistické testy (Mann-Whitney U / Chi²) + efektové veľkosti
  8. Korelácia features s cieľovou premennou
  9. Korelačná matica (P1 features + top korelácie s A10)

Výstupy:
  eda.xlsx  (sheets: missing, testy, korelacia_a10)
  eda_missing.png
  eda_demographics.png
  eda_numeric_boxplots.png
  eda_correlation_target.png
  eda_heatmap_p1.png

Spustenie:
    python eda.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy import stats

RANDOM_STATE = 42

# True  = P1–P7 vypadnú, vznikne Ma_diag_srdcove_ochorenie
# False = P1–P7 ostanú ako samostatné features (bez kompozitu)
USE_MA_DIAG = False

_eda_sheets: dict = {}   # collector — každý df sa uloží tu, na konci → eda.xlsx


# =============================================================
# 1. NAČÍTANIE A PREDSPRACOVANIE  (identické s analyza.py)
# =============================================================

print("=" * 65)
print("1. NAČÍTANIE A PREDSPRACOVANIE")
print("=" * 65)

df = pd.read_csv('data_full1.csv')
_n_raw = len(df)
_n_no_a10 = int(
    (df['A10'].isna() | (df['A10'] == -1)).sum()
) if 'A10' in df.columns else 0

print(f"Načítané záznamy: {_n_raw}  |  stĺpcov: {df.shape[1]}")

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

print(f"Po predspracovaní: {len(y)} pacientov, {df_feat.shape[1]} features")
print(f"A10=1: {y.sum()} ({y.mean()*100:.1f}%)  A10=0: {(1-y).sum()}")

# skupiny pre porovnanie
_pos_idx = np.where(y == 1)[0]
_neg_idx = np.where(y == 0)[0]
df_pos   = df_feat.iloc[_pos_idx]
df_neg   = df_feat.iloc[_neg_idx]


# =============================================================
# 2. ANALÝZA CHÝBAJÚCICH DÁT
# =============================================================

print("\n" + "=" * 65)
print("2. ANALÝZA CHÝBAJÚCICH DÁT")
print("=" * 65)

_miss_pct = df_feat.isnull().mean() * 100
_miss_pct = _miss_pct[_miss_pct > 0].sort_values(ascending=False)
_n_pat_miss = int(df_feat.isnull().any(axis=1).sum())

print(f"  Features s chýbajúcimi hodnotami : {len(_miss_pct)} / {df_feat.shape[1]} "
      f"({len(_miss_pct)/df_feat.shape[1]*100:.1f} %)")
print(f"  Pacienti s aspoň 1 chýbajúcou   : {_n_pat_miss} / {len(df_feat)} "
      f"({_n_pat_miss/len(df_feat)*100:.1f} %)")

if len(_miss_pct) > 0:
    print(f"\n  {'Feature':<35} {'% missing':>10}  {'n':>6}")
    print("  " + "─" * 55)
    for feat, pct in _miss_pct.head(20).items():
        n_m  = int(df_feat[feat].isnull().sum())
        flag = "  !!!" if pct > 20 else ("  !" if pct > 10 else "")
        print(f"  {feat:<35} {pct:>9.1f}%  {n_m:>6}{flag}")
    if len(_miss_pct) > 20:
        print(f"  ... ďalších {len(_miss_pct) - 20} features — pozri sheet 'missing' v eda.xlsx")

    _miss_df = pd.DataFrame({
        'Feature':    _miss_pct.index,
        'pct_missing': _miss_pct.round(2).values,
        'n_missing':  [int(df_feat[f].isnull().sum()) for f in _miss_pct.index],
    })
    _eda_sheets['missing'] = _miss_df
    print("\nPripravené: sheet 'missing'")
    print("  ! >10 %   !!! >20 %  |  Stratégia: imputácia mediánom vnútri CV foldu.")

    # bar chart chýbajúcich dát
    _top_miss = _miss_pct.head(20)
    fig_m, ax_m = plt.subplots(figsize=(10, max(4, len(_top_miss) * 0.35)))
    colors_m = ['#D94F4F' if p > 20 else '#F0A830' if p > 10 else '#5B9BD5'
                for p in _top_miss.values]
    ax_m.barh(_top_miss.index[::-1], _top_miss.values[::-1], color=colors_m[::-1])
    ax_m.axvline(10, color='#F0A830', lw=1.2, linestyle='--', alpha=0.7, label='>10 %')
    ax_m.axvline(20, color='#D94F4F', lw=1.2, linestyle='--', alpha=0.7, label='>20 %')
    ax_m.set_xlabel('% chýbajúcich hodnôt')
    ax_m.set_title('Chýbajúce hodnoty — top features (eda)', fontsize=11)
    ax_m.legend(fontsize=9)
    ax_m.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_missing.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Uložené: eda_missing.png")


# =============================================================
# 3. OPIS DATASETU — DEMOGRAFIA
# =============================================================

print("\n" + "=" * 65)
print("3. OPIS DATASETU — DEMOGRAFIA")
print("=" * 65)

_n   = len(y)
_pos = int(y.sum())
_neg = _n - _pos

print(f"  Záznamy v súbore         : {_n_raw}")
print(f"  Z toho s vyplneným A10   : {_n}  → títo tvoria analytickú vzorku")
print(f"  Vylúčení (A10 chýba)     : {_n_raw - _n}")
print()
print(f"  HUTT pozitívny (A10=1)  : {_pos}  ({_pos/_n*100:.1f} %)")
print(f"  HUTT negatívny (A10=0)  : {_neg}  ({_neg/_n*100:.1f} %)")
print(f"  Pomer tried (neg/pos)   : {_neg/_pos:.2f}")

fig_dem, axes_dem = plt.subplots(1, 3, figsize=(14, 4))
fig_dem.suptitle('Demografický prehľad datasetu (eda)', fontsize=11)

# --- koláčový graf A10
axes_dem[0].pie([_pos, _neg],
                labels=[f'HUTT+\nn={_pos} ({_pos/_n*100:.1f}%)',
                        f'HUTT−\nn={_neg} ({_neg/_n*100:.1f}%)'],
                colors=['#1D9E75', '#888780'],
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 9})
axes_dem[0].set_title('Cieľová premenná A10')

# --- rozloženie veku
if 'Vek' in df_feat.columns:
    _vek   = df_feat['Vek'].dropna()
    _vek_p = df_pos['Vek'].dropna()
    _vek_n = df_neg['Vek'].dropna()
    print(f"\n  Vek: priemer={_vek.mean():.1f}  SD={_vek.std():.1f}  "
          f"medián={_vek.median():.0f}  rozsah=[{_vek.min():.0f}–{_vek.max():.0f}]")
    print(f"  Vek HUTT+: {_vek_p.mean():.1f}±{_vek_p.std():.1f}  "
          f"HUTT−: {_vek_n.mean():.1f}±{_vek_n.std():.1f}")

    axes_dem[1].hist(_vek_p, bins=15, alpha=0.6, color='#1D9E75', label='HUTT+', density=True)
    axes_dem[1].hist(_vek_n, bins=15, alpha=0.6, color='#888780', label='HUTT−', density=True)
    axes_dem[1].axvline(_vek.median(), color='black', lw=1.2, linestyle='--',
                        label=f'Medián={_vek.median():.0f}')
    axes_dem[1].set_xlabel('Vek (roky)')
    axes_dem[1].set_ylabel('Hustota')
    axes_dem[1].set_title('Distribúcia veku')
    axes_dem[1].legend(fontsize=8)
    axes_dem[1].grid(alpha=0.3)

# --- pohlavie
if 'Pohlavie' in df_feat.columns:
    _poh   = df_feat['Pohlavie'].dropna()
    _n_f   = int((_poh == 0).sum())
    _n_m   = int((_poh == 1).sum())
    # prevalencia HUTT+ per pohlavie
    _f_idx = df_feat.index[df_feat['Pohlavie'] == 0]
    _m_idx = df_feat.index[df_feat['Pohlavie'] == 1]
    _f_pos = y[df_feat.index.get_indexer(_f_idx)]
    _m_pos = y[df_feat.index.get_indexer(_m_idx)]
    print(f"\n  Pohlavie: F={_n_f} ({_n_f/_n*100:.1f} %)  M={_n_m} ({_n_m/_n*100:.1f} %)")
    print(f"  Prevalencia HUTT+: F={_f_pos.mean()*100:.1f} %  M={_m_pos.mean()*100:.1f} %")

    _cats   = ['F', 'M']
    _counts = [_n_f, _n_m]
    _prev_p = [_f_pos.mean()*100, _m_pos.mean()*100]
    _x      = np.arange(2)
    _bars   = axes_dem[2].bar(_x, _counts, color=['#C357A5', '#378ADD'],
                              alpha=0.8, width=0.5)
    ax2r    = axes_dem[2].twinx()
    ax2r.plot(_x, _prev_p, 'D--', color='#D94F4F', lw=1.5, ms=7, label='HUTT+ %')
    ax2r.set_ylabel('HUTT+ prevalencia (%)', color='#D94F4F')
    ax2r.tick_params(axis='y', labelcolor='#D94F4F')
    ax2r.set_ylim(0, 100)
    axes_dem[2].set_xticks(_x)
    axes_dem[2].set_xticklabels(_cats)
    axes_dem[2].set_ylabel('Počet pacientov')
    axes_dem[2].set_title('Pohlavie a prevalencia HUTT+')
    axes_dem[2].grid(axis='y', alpha=0.3)
    ax2r.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('eda_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nUložené: eda_demographics.png")


# =============================================================
# 4. IDENTIFIKÁCIA TYPOV FEATURES
# =============================================================

# numerické = viac ako 2 unikátne hodnoty (po imputácii NaN)
_num_feats = [c for c in df_feat.columns
              if df_feat[c].dropna().nunique() > 2]
# binárne = 0/1 (alebo len 2 hodnoty)
_bin_feats = [c for c in df_feat.columns
              if df_feat[c].dropna().nunique() <= 2 and c not in _num_feats]

print(f"\n  Numerické features : {len(_num_feats)}")
print(f"  Binárne features   : {len(_bin_feats)}")


# =============================================================
# 5. NUMERICKÉ FEATURES — DISTRIBÚCIE A POROVNANIE HUTT+/HUTT-
# =============================================================

print("\n" + "=" * 65)
print("5. NUMERICKÉ FEATURES — DISTRIBÚCIE")
print("=" * 65)

if _num_feats:
    _n_cols = min(3, len(_num_feats))
    _n_rows = (len(_num_feats) + _n_cols - 1) // _n_cols
    fig_box, axes_box = plt.subplots(
        _n_rows, _n_cols,
        figsize=(_n_cols * 4, _n_rows * 3.5))
    axes_box = np.array(axes_box).flatten() if _n_rows * _n_cols > 1 else [axes_box]

    for i, feat in enumerate(_num_feats):
        ax = axes_box[i]
        _vp = df_pos[feat].dropna().values
        _vn = df_neg[feat].dropna().values
        ax.boxplot([_vn, _vp], labels=['HUTT−', 'HUTT+'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#DDEEFF'),
                   medianprops=dict(color='#D94F4F', lw=2))
        ax.set_title(feat, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    for j in range(i + 1, len(axes_box)):
        axes_box[j].set_visible(False)

    fig_box.suptitle('Numerické features — HUTT+ vs HUTT− (eda)', fontsize=11)
    plt.tight_layout()
    plt.savefig('eda_numeric_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Uložené: eda_numeric_boxplots.png  ({len(_num_feats)} features)")

    print(f"\n  {'Feature':<20} {'n_pos':>6}  {'med_pos':>8}  "
          f"{'n_neg':>6}  {'med_neg':>8}")
    print("  " + "─" * 55)
    for feat in _num_feats:
        _vp = df_pos[feat].dropna()
        _vn = df_neg[feat].dropna()
        print(f"  {feat:<20} {len(_vp):>6}  {_vp.median():>8.2f}  "
              f"{len(_vn):>6}  {_vn.median():>8.2f}")


# =============================================================
# 6. BINÁRNE FEATURES — FREKVENCIA A ASOCIÁCIA
# =============================================================

print("\n" + "=" * 65)
print("6. BINÁRNE FEATURES — FREKVENCIA")
print("=" * 65)
print(f"  Celkovo binárnych features: {len(_bin_feats)}")
print(f"  (Podrobné testy v sekcii 7 a 8)\n")

# ukážka: prvých 15 binárnych features zoradených podľa frekvencie v HUTT+
_bin_summary = []
for feat in _bin_feats:
    _vp = df_pos[feat].dropna()
    _vn = df_neg[feat].dropna()
    if len(_vp) == 0 or len(_vn) == 0:
        continue
    _bin_summary.append({
        'Feature':    feat,
        'prev_pos%':  round(_vp.mean() * 100, 1),
        'prev_neg%':  round(_vn.mean() * 100, 1),
        'n_pos':      len(_vp),
        'n_neg':      len(_vn),
    })
_bin_df = pd.DataFrame(_bin_summary).sort_values('prev_pos%', ascending=False)
_always_one = _bin_df[(_bin_df['prev_pos%'] == 100.0) & (_bin_df['prev_neg%'] == 100.0)]
_bin_meaningful = _bin_df[~((_bin_df['prev_pos%'] == 100.0) & (_bin_df['prev_neg%'] == 100.0))]

print(f"  Zobrazených top 20 (z {len(_bin_meaningful)} po vylúčení súhrnných "
      f"sekčných atribútov, celkovo {len(_bin_df)} binárnych features):")
print(f"  {'Feature':<35} {'HUTT+%':>7}  {'HUTT−%':>7}")
print("  " + "─" * 52)
for _, row in _bin_meaningful.head(20).iterrows():
    print(f"  {row['Feature']:<35} {row['prev_pos%']:>6.1f}%  {row['prev_neg%']:>6.1f}%")

# Identifikuj features so 100 % v oboch skupinách
_always_one = _bin_df[(_bin_df['prev_pos%'] == 100.0) & (_bin_df['prev_neg%'] == 100.0)]
if len(_always_one) > 0:
    print(f"\n  POZNÁMKA — súhrnné sekčné atribúty ({len(_always_one)} features):")
    print(f"  {', '.join(_always_one['Feature'].tolist())}")
    print(f"""
  Tieto stĺpce sú automaticky generované z dotazníka ako sekčné príznaky.
  Napríklad stĺpec 'B' = 1 ak pacient odpovedal aspoň na jednu otázku
  zo sekcie B dotazníka (B1, B2, ...). Keďže každý pacient vyplnil
  dotazník, všetci majú hodnotu 1 → HUTT+% aj HUTT−% = 100 %.
  Nenesú žiadnu klinickú informáciu a ConsensusFS ich automaticky
  vyradí (Chi² test bude nevýznamný, RF importance = 0).
""")


# =============================================================
# 7. ŠTATISTICKÉ TESTY
# =============================================================

print("\n" + "=" * 65)
print("7. ŠTATISTICKÉ TESTY  (Mann-Whitney U / Chi²)")
print("=" * 65)
print("  Mann-Whitney U pre numerické features (bez predpokladu normality)")
print("  Chi² pre binárne features")
print("  Efektová veľkosť: r = Z/sqrt(N) pre MWU  |  Phi pre Chi²")
print()

test_rows = []

# numerické — Mann-Whitney U
for feat in _num_feats:
    _vp = df_pos[feat].dropna().values
    _vn = df_neg[feat].dropna().values
    if len(_vp) < 5 or len(_vn) < 5:
        continue
    stat, pval = stats.mannwhitneyu(_vp, _vn, alternative='two-sided')
    # rank-biserial correlation ako efektová veľkosť
    n1, n2 = len(_vp), len(_vn)
    r_rb   = 1 - (2 * stat) / (n1 * n2)
    sig    = ("***" if pval < 0.001 else "**" if pval < 0.01
              else "*" if pval < 0.05 else "n.s.")
    test_rows.append({
        'Feature': feat, 'Test': 'Mann-Whitney U',
        'Statistic': round(stat, 2), 'p_value': round(pval, 4),
        'Effect_size': round(abs(r_rb), 4), 'Significance': sig,
        'Median_pos': round(np.median(_vp), 3),
        'Median_neg': round(np.median(_vn), 3),
    })

# binárne — Chi²
for feat in _bin_feats:
    _col = df_feat[feat].dropna()
    _y_s = y[df_feat[feat].notna().values]
    if len(_col) < 10:
        continue
    ct = pd.crosstab(_col, _y_s)
    if ct.shape != (2, 2):
        continue
    chi2, pval, _, _ = stats.chi2_contingency(ct, correction=False)
    phi = np.sqrt(chi2 / len(_col))
    sig = ("***" if pval < 0.001 else "**" if pval < 0.01
           else "*" if pval < 0.05 else "n.s.")
    test_rows.append({
        'Feature': feat, 'Test': 'Chi²',
        'Statistic': round(chi2, 3), 'p_value': round(pval, 4),
        'Effect_size': round(phi, 4), 'Significance': sig,
        'Median_pos': None, 'Median_neg': None,
    })

tests_df = pd.DataFrame(test_rows).sort_values('p_value')
_eda_sheets['testy'] = tests_df
print("Pripravené: sheet 'testy'")

# výpis signifikantných
sig_df = tests_df[tests_df['Significance'] != 'n.s.']
print(f"\n  Signifikantných features (p<0.05): {len(sig_df)} / {len(test_rows)}")
print(f"\n  {'Feature':<35} {'Test':<16} {'p-value':>8}  {'Efekt':>6}  {'Sig':>5}")
print("  " + "─" * 75)
for _, row in sig_df.head(30).iterrows():
    print(f"  {row['Feature']:<35} {row['Test']:<16} "
          f"{row['p_value']:>8.4f}  {row['Effect_size']:>6.4f}  {row['Significance']:>5}")


# =============================================================
# 8. KORELÁCIA FEATURES S CIEĽOVOU PREMENNOU
# =============================================================

print("\n" + "=" * 65)
print("8. KORELÁCIA S CIEĽOVOU PREMENNOU (A10)")
print("=" * 65)
print("  Pearson/Phi korelácia každej feature s A10 (binárna cieľová premenná)")
print()

corr_rows = []
for feat in df_feat.columns:
    _col = df_feat[feat]
    _mask = _col.notna()
    if _mask.sum() < 20:
        continue
    r, pval = stats.pearsonr(_col[_mask].values, y[_mask.values])
    corr_rows.append({
        'Feature':     feat,
        'Correlation': round(r, 4),
        'Abs_corr':    round(abs(r), 4),
        'p_value':     round(pval, 4),
        'Significant': pval < 0.05,
    })

corr_df = pd.DataFrame(corr_rows).sort_values('Abs_corr', ascending=False)
_eda_sheets['korelacia_a10'] = corr_df
print("Pripravené: sheet 'korelacia_a10'")

print(f"\n  Top 20 z {len(corr_df)} features podľa |korelácie| s A10 (všetky v eda.xlsx):")
print(f"  {'Feature':<35} {'r':>7}  {'p-value':>8}  {'Sig':>5}")
print("  " + "─" * 60)
for _, row in corr_df.head(20).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 \
          else "*" if row['p_value'] < 0.05 else "n.s."
    print(f"  {row['Feature']:<35} {row['Correlation']:>7.4f}  "
          f"{row['p_value']:>8.4f}  {sig:>5}")

# bar chart top korelácie
_top_corr = corr_df.head(20).sort_values('Correlation')
_colors_c = ['#D94F4F' if v > 0 else '#378ADD' for v in _top_corr['Correlation']]
fig_c, ax_c = plt.subplots(figsize=(9, max(5, len(_top_corr) * 0.4)))
ax_c.barh(_top_corr['Feature'], _top_corr['Correlation'],
          color=_colors_c, alpha=0.85)
ax_c.axvline(0, color='black', lw=0.8)
ax_c.set_xlabel('Pearson / Phi korelácia s A10')
ax_c.set_title('Top 20 features — korelácia s HUTT výsledkom (eda)', fontsize=11)
ax_c.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_correlation_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("Uložené: eda_correlation_target.png")


# =============================================================
# 9. KORELAČNÁ MATICA
# =============================================================

print("\n" + "=" * 65)
print("9. KORELAČNÁ MATICA")
print("=" * 65)

# P1 features — malá, čitateľná heatmapa
P1_COLS = [c for c in ['A2_sys', 'A2_dia', 'A3', 'Vek', 'Pohlavie']
           if c in df_feat.columns]
if len(P1_COLS) >= 2:
    _cm_p1 = df_feat[P1_COLS].corr()
    fig_h1, ax_h1 = plt.subplots(figsize=(6, 5))
    _im = ax_h1.imshow(_cm_p1.values, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(_im, ax=ax_h1, shrink=0.8)
    ax_h1.set_xticks(range(len(P1_COLS)))
    ax_h1.set_yticks(range(len(P1_COLS)))
    ax_h1.set_xticklabels(P1_COLS, rotation=45, ha='right', fontsize=9)
    ax_h1.set_yticklabels(P1_COLS, fontsize=9)
    for i in range(len(P1_COLS)):
        for j in range(len(P1_COLS)):
            ax_h1.text(j, i, f"{_cm_p1.values[i, j]:.2f}",
                       ha='center', va='center', fontsize=8,
                       color='white' if abs(_cm_p1.values[i, j]) > 0.5 else 'black')
    ax_h1.set_title('Korelačná matica — P1 features (eda)', fontsize=11)
    plt.tight_layout()
    plt.savefig('eda_heatmap_p1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Uložené: eda_heatmap_p1.png  (P1 anamnézové features)")

# top 15 features podľa korelácie s A10 — heatmapa vzájomných korelácií
_top15 = corr_df.head(15)['Feature'].tolist()
_top15 = [f for f in _top15 if f in df_feat.columns]
if len(_top15) >= 2:
    _cm_top = df_feat[_top15].corr()
    fig_h2, ax_h2 = plt.subplots(figsize=(10, 8))
    _im2 = ax_h2.imshow(_cm_top.values, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(_im2, ax=ax_h2, shrink=0.8)
    ax_h2.set_xticks(range(len(_top15)))
    ax_h2.set_yticks(range(len(_top15)))
    ax_h2.set_xticklabels(_top15, rotation=45, ha='right', fontsize=8)
    ax_h2.set_yticklabels(_top15, fontsize=8)
    for i in range(len(_top15)):
        for j in range(len(_top15)):
            _v = _cm_top.values[i, j]
            ax_h2.text(j, i, f"{_v:.2f}", ha='center', va='center',
                       fontsize=7,
                       color='white' if abs(_v) > 0.5 else 'black')
    ax_h2.set_title('Korelačná matica — top 15 features (podľa |r| s A10)', fontsize=11)
    plt.tight_layout()
    plt.savefig('eda_heatmap_top15.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Uložené: eda_heatmap_top15.png  (top 15 podľa |r| s A10)")


# =============================================================
# ZHRNUTIE EDA
# =============================================================

print("\n" + "=" * 65)
print("ZHRNUTIE EDA")
print("=" * 65)
print(f"  Dataset       : {_n} pacientov  |  HUTT+={_pos} ({_pos/_n*100:.1f}%)  "
      f"HUTT−={_neg} ({_neg/_n*100:.1f}%)")
print(f"  Features      : {df_feat.shape[1]} celkom  "
      f"({len(_num_feats)} numerické, {len(_bin_feats)} binárne)")
print(f"  Chýbajúce     : {len(_miss_pct)} features s NaN  |  "
      f"{_n_pat_miss} pacientov ({_n_pat_miss/_n*100:.1f} %) má aspoň 1 NaN")
_sig_count = len(tests_df[tests_df['Significance'] != 'n.s.'])
print(f"  Signifikantné : {_sig_count} / {len(test_rows)} features (p<0.05 po teste)")
print(f"  Top korelácia s A10: {corr_df.iloc[0]['Feature']}  "
      f"r={corr_df.iloc[0]['Correlation']:.4f}")
print()

# =============================================================
# ULOŽENIE DO EXCELU  (sheets: missing | testy | korelacia_a10)
# =============================================================

_EXCEL_EDA = 'eda.xlsx'
with pd.ExcelWriter(_EXCEL_EDA, engine='openpyxl') as _writer:
    for _sheet, _df in _eda_sheets.items():
        _df.to_excel(_writer, sheet_name=_sheet, index=False)
print(f"Uložené: {_EXCEL_EDA}  (sheets: {', '.join(_eda_sheets.keys())})")

print("  Výstupné súbory:")
print(f"    {_EXCEL_EDA}  (sheets: missing, testy, korelacia_a10)")
print("    eda_missing.png")
print("    eda_demographics.png")
print("    eda_numeric_boxplots.png")
print("    eda_correlation_target.png")
print("    eda_heatmap_p1.png")
print("    eda_heatmap_top15.png")
