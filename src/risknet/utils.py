import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
CONFIG = {'risk_thresholds': [0.3, 0.6, 0.8], 'decision_threshold': 0.2,
          'levels': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], 'priorities': ['P4', 'P3', 'P2', 'P1']}

def save_json(path, value):
    def sanitize(x):
        if isinstance(x, dict): return {str(k): sanitize(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [sanitize(v) for v in x]
        if isinstance(x, np.generic): x = x.item()
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sanitize(value), indent=2, default=str, allow_nan=False), encoding='utf-8')

def risk_mapping(probability, config=CONFIG):
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Probability must be finite and between zero and one')
    index = sum(probability >= t for t in config['risk_thresholds'])
    return config['levels'][index], config['priorities'][index]
