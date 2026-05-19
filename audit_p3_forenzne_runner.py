
import json
import os
import joblib
import pandas as pd
import numpy as np

try:
    from model_components import ConsensusFeatureSelector, P3SelectorConsensus  # noqa: F401
except Exception:
    pass

MODEL_PATH = 'model_p3_et.joblib'
OUT_JSON = 'audit_p3_forenzne_report.json'
OUT_CSV = 'audit_p3_forenzne_diff.csv'
OUT_MD = 'audit_p3_forenzne_report.md'

pkg = joblib.load(MODEL_PATH)
pipe = pkg['pipeline']
features = list(pkg.get('features', []))
selected_features = list(pkg.get('selected_features', []))
p2_selected_features = list(pkg.get('p2_selected_features', []))
cv_consensus_features = list(pkg.get('cv_consensus_features', []))
threshold = pkg.get('threshold', None)
model_name = pkg.get('model_name', None)

step_names = [name for name, _ in pipe.steps]
fs = pipe.named_steps.get('fs', None)
clf = pipe.named_steps.get('clf', None)

runtime_support_names = []
runtime_p1_passthrough = []
runtime_p2_selected = []
runtime_has_support = False
n_p1 = None

if fs is not None and hasattr(fs, 'get_support'):
    mask = np.asarray(fs.get_support(), dtype=bool)
    runtime_has_support = True
    if len(mask) == len(features):
        runtime_support_names = [f for f, m in zip(features, mask) if m]
    if hasattr(fs, 'n_p1'):
        n_p1 = int(fs.n_p1)
        runtime_p1_passthrough = features[:n_p1]
        if len(mask) == len(features):
            runtime_p2_selected = [f for f, m in zip(features[n_p1:], mask[n_p1:]) if m]

records = []
all_names = sorted(set(features) | set(selected_features) | set(p2_selected_features) | set(cv_consensus_features) | set(runtime_support_names) | set(runtime_p2_selected))
for name in all_names:
    records.append({
        'feature': name,
        'in_features_pool': name in features,
        'in_selected_features': name in selected_features,
        'in_p2_selected_features': name in p2_selected_features,
        'in_cv_consensus_features': name in cv_consensus_features,
        'in_runtime_support': name in runtime_support_names,
        'in_runtime_p2_selected': name in runtime_p2_selected,
        'is_p1_passthrough': name in runtime_p1_passthrough,
    })

df = pd.DataFrame(records).sort_values('feature')
df.to_csv(OUT_CSV, index=False)

summary = {
    'model_path': MODEL_PATH,
    'model_name': model_name,
    'threshold': threshold,
    'pipeline_steps': step_names,
    'n_features_pool': len(features),
    'n_selected_features_metadata': len(selected_features),
    'n_p2_selected_features_metadata': len(p2_selected_features),
    'n_cv_consensus_features': len(cv_consensus_features),
    'runtime_has_support': runtime_has_support,
    'runtime_n_support': len(runtime_support_names),
    'runtime_n_p1_passthrough': len(runtime_p1_passthrough),
    'runtime_n_p2_selected': len(runtime_p2_selected),
    'n_p1_from_fs': n_p1,
    'features_pool': features,
    'selected_features_metadata': selected_features,
    'p2_selected_features_metadata': p2_selected_features,
    'cv_consensus_features': cv_consensus_features,
    'runtime_support_names': runtime_support_names,
    'runtime_p1_passthrough': runtime_p1_passthrough,
    'runtime_p2_selected': runtime_p2_selected,
    'diffs': {
        'selected_minus_runtime': sorted(set(selected_features) - set(runtime_support_names)),
        'runtime_minus_selected': sorted(set(runtime_support_names) - set(selected_features)),
        'p2meta_minus_runtime_p2': sorted(set(p2_selected_features) - set(runtime_p2_selected)),
        'runtime_p2_minus_p2meta': sorted(set(runtime_p2_selected) - set(p2_selected_features)),
        'cv_minus_runtime': sorted(set(cv_consensus_features) - set(runtime_support_names)),
        'runtime_minus_cv': sorted(set(runtime_support_names) - set(cv_consensus_features)),
    }
}

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

md = []
md.append('# Forenzný audit P3 feature selection')
md.append('')
md.append(f"- Model: {model_name}")
md.append(f"- Threshold: {threshold}")
md.append(f"- Pipeline kroky: {', '.join(step_names)}")
md.append(f"- Features pool: {len(features)}")
md.append(f"- selected_features metadata: {len(selected_features)}")
md.append(f"- p2_selected_features metadata: {len(p2_selected_features)}")
md.append(f"- cv_consensus_features: {len(cv_consensus_features)}")
md.append(f"- runtime support: {len(runtime_support_names)}")
md.append(f"- runtime P1 passthrough: {len(runtime_p1_passthrough)}")
md.append(f"- runtime P2 selected: {len(runtime_p2_selected)}")
md.append('')
md.append('## Najdôležitejšie rozdiely')
md.append(f"- selected_minus_runtime: {summary['diffs']['selected_minus_runtime']}")
md.append(f"- runtime_minus_selected: {summary['diffs']['runtime_minus_selected']}")
md.append(f"- p2meta_minus_runtime_p2: {summary['diffs']['p2meta_minus_runtime_p2']}")
md.append(f"- runtime_p2_minus_p2meta: {summary['diffs']['runtime_p2_minus_p2meta']}")
md.append('')
md.append('## Runtime P2 selected')
md.append(', '.join(runtime_p2_selected) if runtime_p2_selected else '(none)')
md.append('')
md.append('## Metadata P2 selected')
md.append(', '.join(p2_selected_features) if p2_selected_features else '(none)')

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print('Vytvorené:', OUT_JSON, OUT_CSV, OUT_MD)
