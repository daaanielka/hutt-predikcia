import os
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)

REPORT = {
    'workspace_files_checked': [],
    'model_files_found': [],
    'issues': [],
    'notes': [],
    'packages': {},
    'app_alignment': {},
    'scenario_tests': {},
}


def add_issue(msg):
    REPORT['issues'].append(msg)


def add_note(msg):
    REPORT['notes'].append(msg)


# -----------------------------
# Try importing app/analyza code
# -----------------------------
try:
    import joblib
except Exception as e:
    raise RuntimeError(f'joblib import failed: {e}')

for fname in ['app.py', 'analyza.py', 'validacia.py', 'requirements.txt', 'README.md']:
    p = ROOT / fname
    if p.exists():
        REPORT['workspace_files_checked'].append(str(p.name))

app_text = (ROOT / 'app.py').read_text(encoding='utf-8', errors='ignore') if (ROOT / 'app.py').exists() else ''
analyza_text = (ROOT / 'analyza.py').read_text(encoding='utf-8', errors='ignore') if (ROOT / 'analyza.py').exists() else ''
validacia_text = (ROOT / 'validacia.py').read_text(encoding='utf-8', errors='ignore') if (ROOT / 'validacia.py').exists() else ''

# -----------------------------
# Find model artifacts
# -----------------------------
model_candidates = sorted([p for p in ROOT.rglob('*.joblib') if p.is_file()])
REPORT['model_files_found'] = [str(p.relative_to(ROOT)) for p in model_candidates]
if not model_candidates:
    add_issue('Nenašli sa žiadne .joblib modelové súbory v workspace. Audit uložených modelov nebude možné dokončiť bez artefaktov.')

# -----------------------------
# App alignment inspection
# -----------------------------
REPORT['app_alignment'] = {
    'requires_threshold_key': '"threshold"' in app_text or "'threshold'" in app_text,
    'hardcoded_threshold_p1_05': 'THRESHOLD_P1 = 0.5' in app_text,
    'hardcoded_threshold_p3_05': 'THRESHOLD_P3 = 0.5' in app_text,
    'reads_pkg_features': 'pkg_p1["features"]' in app_text and 'pkg_p3["features"]' in app_text,
    'reads_pkg_p2_selected_features': 'pkg_p3["p2_selected_features"]' in app_text,
    'uses_pkg_threshold_in_score_card_display': 'pkg.get("threshold", 0.5)' in app_text,
}
if REPORT['app_alignment']['requires_threshold_key'] and (
    REPORT['app_alignment']['hardcoded_threshold_p1_05'] or REPORT['app_alignment']['hardcoded_threshold_p3_05']
):
    add_issue('app.py vyžaduje threshold v package metadata, ale zároveň používa hardcoded THRESHOLD_P1/THRESHOLD_P3 = 0.5; treba zjednotiť policy.')

# -----------------------------
# Analyze packages
# -----------------------------
rows_overview = []
rows_features = []
rows_steps = []
rows_selected = []
rows_scenarios = []

scenario_inputs = {
    'baseline_mid': {
        'A2_sys': 120, 'A2_dia': 80, 'A3': 72, 'Vek': 40, 'Pohlavie': 0,
        'C1': 20, 'C2': 3, 'C4': 25,
        'D2': 0, 'E2': 0, 'E3': 0, 'E4': 0, 'E5': 0,
        'H2': 0, 'H3': 0, 'H4': 0, 'H6': 0, 'N6': 1,
        'P4': 0, 'P11': 0, 'P12': 0, 'P13': 0,
    },
    'vvs_like': {
        'A2_sys': 105, 'A2_dia': 68, 'A3': 78, 'Vek': 24, 'Pohlavie': 0,
        'C1': 16, 'C2': 6, 'C4': 20,
        'D2': 1, 'E2': 1, 'E3': 1, 'E4': 1, 'E5': 1,
        'H2': 1, 'H3': 1, 'H4': 1, 'H6': 1, 'N6': 1,
        'P4': 0, 'P11': 1, 'P12': 1, 'P13': 0,
    },
    'alarm_like': {
        'A2_sys': 150, 'A2_dia': 95, 'A3': 110, 'Vek': 72, 'Pohlavie': 1,
        'C1': 70, 'C2': 1, 'C4': 72,
        'D2': 0, 'E2': 0, 'E3': 0, 'E4': 0, 'E5': 0,
        'H2': 0, 'H3': 0, 'H4': 0, 'H6': 0, 'N6': 0,
        'P4': 1, 'P11': 0, 'P12': 0, 'P13': 1,
    },
    'missing_heavy': {
        'A2_sys': np.nan, 'A2_dia': np.nan, 'A3': np.nan, 'Vek': 55, 'Pohlavie': 1,
        'C1': np.nan, 'C2': np.nan, 'C4': np.nan,
        'D2': np.nan, 'E2': np.nan, 'E3': np.nan, 'E4': np.nan, 'E5': np.nan,
        'H2': np.nan, 'H3': np.nan, 'H4': np.nan, 'H6': np.nan, 'N6': np.nan,
        'P4': np.nan, 'P11': np.nan, 'P12': np.nan, 'P13': np.nan,
    }
}

for model_path in model_candidates:
    rel = str(model_path.relative_to(ROOT))
    try:
        pkg = joblib.load(model_path)
    except Exception as e:
        add_issue(f'Nepodarilo sa načítať {rel}: {e}')
        continue

    pkg_name = model_path.stem
    REPORT['packages'][pkg_name] = {
        'path': rel,
        'keys': sorted(list(pkg.keys())) if isinstance(pkg, dict) else None,
    }

    if not isinstance(pkg, dict):
        add_issue(f'{rel} nie je uložený ako dict package; app.py očakáva dict s metadata.')
        continue

    pipeline = pkg.get('pipeline')
    features = list(pkg.get('features', [])) if isinstance(pkg.get('features', []), (list, tuple)) else []
    selected = list(pkg.get('selected_features', [])) if isinstance(pkg.get('selected_features', []), (list, tuple)) else []
    p2_selected = list(pkg.get('p2_selected_features', [])) if isinstance(pkg.get('p2_selected_features', []), (list, tuple)) else []
    threshold = pkg.get('threshold', None)

    step_names = []
    if pipeline is not None and hasattr(pipeline, 'steps'):
        step_names = [name for name, _ in pipeline.steps]
        for i, (step_name, step_obj) in enumerate(pipeline.steps):
            rows_steps.append({
                'model_file': rel,
                'step_order': i + 1,
                'step_name': step_name,
                'step_class': step_obj.__class__.__name__,
            })

    has_fs = 'fs' in step_names
    has_scaler = 'scaler' in step_names
    has_imp = 'imp' in step_names
    has_clf = 'clf' in step_names

    rows_overview.append({
        'model_file': rel,
        'model_name': pkg.get('model_name'),
        'threshold': threshold,
        'n_features_pool': len(features),
        'n_selected_features': len(selected),
        'n_p2_selected_features': len(p2_selected),
        'has_pipeline': pipeline is not None,
        'has_imp': has_imp,
        'has_fs': has_fs,
        'has_scaler': has_scaler,
        'has_clf': has_clf,
        'auc_cv_mean': pkg.get('AUC_CV_mean'),
        'auc_std': pkg.get('AUC_std'),
        'sens_at_thr': pkg.get('sensitivity_at_thr'),
        'spec_at_thr': pkg.get('specificity_at_thr'),
        'ppv_at_thr': pkg.get('ppv_at_thr'),
        'npv_at_thr': pkg.get('npv_at_thr'),
    })

    for f in features:
        rows_features.append({'model_file': rel, 'kind': 'features_pool', 'feature': f})
    for f in selected:
        rows_features.append({'model_file': rel, 'kind': 'selected_features', 'feature': f})
    for f in p2_selected:
        rows_features.append({'model_file': rel, 'kind': 'p2_selected_features', 'feature': f})

    feature_union = []
    for source_name, seq in [('features_pool', features), ('selected_features', selected), ('p2_selected_features', p2_selected)]:
        for feat in seq:
            rows_selected.append({
                'model_file': rel,
                'feature': feat,
                'in_features_pool': feat in features,
                'in_selected_features': feat in selected,
                'in_p2_selected_features': feat in p2_selected,
                'source_list': source_name,
            })
            feature_union.append(feat)

    if threshold is None:
        add_issue(f'{rel} nemá threshold metadata.')
    if not has_clf:
        add_issue(f'{rel} pipeline nemá clf krok.')
    if not has_imp:
        add_issue(f'{rel} pipeline nemá imp krok.')
    if 'p3' in pkg_name.lower() and not p2_selected:
        add_issue(f'{rel} vyzerá ako P3 model, ale nemá p2_selected_features.')
    if selected and not set(selected).issubset(set(features)):
        add_issue(f'{rel} selected_features nie sú podmnožinou features poolu.')
    if p2_selected and not set(p2_selected).issubset(set(features)):
        add_issue(f'{rel} p2_selected_features nie sú podmnožinou features poolu.')

    # Scenario tests only if pipeline available
    if pipeline is not None and features:
        for scenario_name, overrides in scenario_inputs.items():
            row = {f: np.nan for f in features}
            for k, v in overrides.items():
                if k in row:
                    row[k] = v
            X = pd.DataFrame([row], columns=features)
            try:
                proba = float(pipeline.predict_proba(X)[:, 1][0])
                pred = int(proba >= float(threshold if threshold is not None else 0.5))
            except Exception as e:
                proba = np.nan
                pred = np.nan
                add_issue(f'Scenario test zlyhal pre {rel} / {scenario_name}: {e}')

            selected_runtime = None
            try:
                if has_fs and hasattr(pipeline.named_steps['fs'], 'get_support'):
                    # fit must already exist in loaded package; if not, this may fail
                    mask = pipeline.named_steps['fs'].get_support()
                    selected_runtime = [f for f, m in zip(features, mask) if bool(m)]
                else:
                    selected_runtime = list(features)
            except Exception:
                selected_runtime = None

            rows_scenarios.append({
                'model_file': rel,
                'scenario': scenario_name,
                'predicted_probability': proba,
                'predicted_class_at_pkg_threshold': pred,
                'threshold_used': float(threshold if threshold is not None else 0.5),
                'runtime_selected_features_count': len(selected_runtime) if selected_runtime is not None else None,
                'runtime_selected_features': ', '.join(selected_runtime) if selected_runtime else None,
            })

# -----------------------------
# Build summary CSV/XLSX/MD
# -----------------------------
overview_df = pd.DataFrame(rows_overview)
features_df = pd.DataFrame(rows_features)
steps_df = pd.DataFrame(rows_steps)
selected_df = pd.DataFrame(rows_selected).drop_duplicates() if rows_selected else pd.DataFrame()
scenarios_df = pd.DataFrame(rows_scenarios)
issues_df = pd.DataFrame({'issue': REPORT['issues']})
notes_df = pd.DataFrame({'note': REPORT['notes']})
app_df = pd.DataFrame([REPORT['app_alignment']])

with pd.ExcelWriter(OUT / 'audit_modelu.xlsx', engine='openpyxl') as writer:
    overview_df.to_excel(writer, sheet_name='overview', index=False)
    features_df.to_excel(writer, sheet_name='features', index=False)
    steps_df.to_excel(writer, sheet_name='pipeline_steps', index=False)
    selected_df.to_excel(writer, sheet_name='feature_membership', index=False)
    scenarios_df.to_excel(writer, sheet_name='scenario_tests', index=False)
    app_df.to_excel(writer, sheet_name='app_alignment', index=False)
    issues_df.to_excel(writer, sheet_name='issues', index=False)
    notes_df.to_excel(writer, sheet_name='notes', index=False)

overview_df.to_csv(OUT / 'audit_modelu_overview.csv', index=False)
scenarios_df.to_csv(OUT / 'audit_modelu_scenarios.csv', index=False)
issues_df.to_csv(OUT / 'audit_modelu_issues.csv', index=False)

md = []
md.append('# Audit modelu\n')
md.append('## Čo skript kontroluje\n')
md.append('- štruktúru uložených joblib balíkov\n')
md.append('- zhodu medzi package metadata a tým, čo číta app.py\n')
md.append('- pipeline kroky (imp / fs / scaler / clf)\n')
md.append('- rozdiel medzi features pool, selected_features a p2_selected_features\n')
md.append('- orientačné scenárové predikcie pre audit správania modelu\n')
md.append('\n## Kľúčové zistenia\n')
if REPORT['issues']:
    for issue in REPORT['issues']:
        md.append(f'- {issue}\n')
else:
    md.append('- Neboli zachytené žiadne hard fail nálezy v štruktúre artefaktov.\n')

if not model_candidates:
    md.append('\n## Dôležité\n')
    md.append('- V pracovnom priestore nebol nájdený žiaden `.joblib` model, preto audit overil iba kódové väzby v `app.py`, `analyza.py` a `validacia.py`, nie skutočne uložené modelové balíky.\n')

md.append('\n## App alignment\n')
for k, v in REPORT['app_alignment'].items():
    md.append(f'- {k}: {v}\n')

(OUT / 'audit_modelu_report.md').write_text(''.join(md), encoding='utf-8')
(OUT / 'audit_modelu_report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding='utf-8')

print('Vygenerované súbory:')
for p in [
    OUT / 'audit_modelu.py',
    OUT / 'audit_modelu.xlsx',
    OUT / 'audit_modelu_overview.csv',
    OUT / 'audit_modelu_scenarios.csv',
    OUT / 'audit_modelu_issues.csv',
    OUT / 'audit_modelu_report.md',
    OUT / 'audit_modelu_report.json',
]:
    print('-', p)
