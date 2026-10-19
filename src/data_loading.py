from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
UPLOADED_PATH = RAW / 'Diabetes Dataset(1).csv'

def load_uploaded(path=UPLOADED_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Uploaded dataset not found: {path}')
    frame = pd.read_csv(path)
    if 'Outcome' not in frame.columns:
        raise ValueError(f"Expected 'Outcome'; found {list(frame.columns)}")
    return frame

def load_cdc(cache=True):
    from ucimlrepo import fetch_ucirepo
    print('Fetching CDC Diabetes Health Indicators (UCI dataset 891)...')
    dataset = fetch_ucirepo(id=891)
    frame = pd.concat([dataset.data.features.reset_index(drop=True), dataset.data.targets.reset_index(drop=True)], axis=1)
    if not any(c in frame.columns for c in ('Diabetes_binary', 'Diabetes_012', 'Outcome')):
        raise ValueError(f'Could not identify target in columns: {list(frame.columns)}')
    if cache:
        RAW.mkdir(parents=True, exist_ok=True)
        frame.to_csv(RAW / 'cdc_diabetes_health_indicators.csv', index=False)
    return frame

def find_target(frame):
    for name in ('Outcome', 'Diabetes_binary', 'Diabetes_012'):
        if name in frame.columns: return name
    raise ValueError('No recognised target column')
