from pathlib import Path
import joblib
import model_components

ROOT = Path(__file__).resolve().parent
MODEL_FILES = ['model_p1_et.joblib', 'model_p3_et.joblib']

for name in MODEL_FILES:
    path = ROOT / name
    if not path.exists():
        print(f'[SKIP] {name} neexistuje')
        continue
    pkg = joblib.load(path)
    backup = ROOT / f'{path.stem}.backup.joblib'
    joblib.dump(pkg, backup)
    joblib.dump(pkg, path)
    print(f'[OK] Re-exported {name} | backup: {backup.name}')
