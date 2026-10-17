import json
from .config import REPORTS

def require_reviewed_gate(explicit_user_approval=False):
    path=REPORTS/'metrics/pretraining_gate.json'
    if not path.is_file(): raise RuntimeError('Run v2 preprocessing and baselines before GPU training')
    gate=json.loads(path.read_text(encoding='utf-8'))
    if not gate['passed']: raise RuntimeError('V2 pre-training gate failed: '+ '; '.join(gate['failures']))
    if not explicit_user_approval:
        raise RuntimeError('V2 gate passed, but GPU training is paused for explicit user review')
    return gate
