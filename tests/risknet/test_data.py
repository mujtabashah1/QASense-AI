"""Audit checks that can run independently of native ML dependencies."""
import hashlib
import json
import pandas as pd
from risknet.data_loader import FEATURES, normalize_schema, feature_groups
from risknet.utils import ROOT

def test_raw_files_unchanged():
    audit=json.loads((ROOT/'reports/risknet/metrics/data_audit.json').read_text())
    for name in ['cm1','jm1','kc1','kc2','pc1']:
        path=ROOT/f'datasets/NASA-promise-dataset-repository-main/{name}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit[name]['sha256']

def test_clean_schema_and_count():
    df=pd.read_csv(ROOT/'datasets/processed/risknet/nasa_combined_clean.csv')
    assert len(df)==17442
    assert list(df.columns)==FEATURES+['defect','project']
    assert set(df.defect)=={0,1}
    assert set(df.project)=={'CM1','JM1','KC1','KC2','PC1'}

def test_duplicate_missing_values_are_grouped():
    df=pd.DataFrame([{c:float('nan') for c in FEATURES}]*2)
    assert feature_groups(df).nunique()==1
