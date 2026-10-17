import numpy as np
import pandas as pd
import pytest
from risknet.data_loader import FEATURES, normalize_schema, feature_groups
from risknet.inference import RiskNet
from risknet.utils import ROOT, risk_mapping

def sample(): return {k: 2.0 for k in FEATURES}

@pytest.mark.parametrize('label,expected',[('yes',1),('no',0),(True,1),(False,0)])
def test_schema(label,expected):
    d = sample(); d['iv(G)'] = d.pop('iv(g)'); d['N'] = d.pop('n'); d['lOCodeAndComment'] = d.pop('locCodeAndComment'); d['problems'] = label
    out = normalize_schema(pd.DataFrame([d]))
    assert list(out.columns) == FEATURES+['defect'] and out.defect.iloc[0] == expected

def test_missing_and_collision():
    with pytest.raises(ValueError): normalize_schema(pd.DataFrame([sample()]))
    d=sample(); d.update(defects=False,N=1)
    with pytest.raises(ValueError): normalize_schema(pd.DataFrame([d]))

def test_invalid_training_tokens_and_target():
    d=sample(); d.update(defects='true',loc='?',v=float('inf'))
    result=normalize_schema(pd.DataFrame([d]))
    assert pd.isna(result.loc[0,'loc']) and pd.isna(result.loc[0,'v'])
    d['defects']='unknown'
    with pytest.raises(ValueError): normalize_schema(pd.DataFrame([d]))

def test_invalid_probability():
    for p in [float('nan'),float('inf'),-1,2]:
        with pytest.raises(ValueError): risk_mapping(p)

@pytest.mark.parametrize('p,level',[(0,'LOW'),(.299,'LOW'),(.3,'MEDIUM'),(.6,'HIGH'),(.8,'CRITICAL'),(1,'CRITICAL')])
def test_risk(p,level): assert risk_mapping(p)[0] == level

def test_groups():
    df = pd.DataFrame([sample(),sample()]); df.loc[1,'loc']=3
    assert feature_groups(df).nunique()==2

def test_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError): RiskNet(tmp_path/'none.joblib')

@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),-1,'2',True])
def test_invalid(value):
    d=sample(); d['loc']=value
    with pytest.raises(ValueError): RiskNet().predict_risk(d)

def test_inference_order():
    model=RiskNet(); d=sample()
    result=model.predict_risk(d)
    assert result == model.predict_risk(dict(reversed(list(d.items()))))
    assert 0 <= result['defect_probability'] <= 1
    assert result['qa_priority'] in ['P1','P2','P3','P4']
    assert result['prediction'] in ['Defect-Prone','Non-Defect-Prone']
    d.pop('loc')
    with pytest.raises(ValueError): model.predict_risk(d)

def test_saved_split_leakage():
    import json
    df=pd.read_csv(ROOT/'datasets/processed/risknet/nasa_combined_clean.csv'); groups=feature_groups(df)
    for split in json.loads((ROOT/'reports/risknet/metrics/split_manifest.json').read_text()):
        assert not set(groups.iloc[split['train_indices']]) & set(groups.iloc[split['test_indices']])
        if split['protocol']=='cross': assert split['project'] not in set(df.iloc[split['train_indices']].project)

def test_local_logistic_explanation():
    model=RiskNet()
    if model.bundle['metadata']['model_type'] != 'logistic': return
    calibrated=model.bundle['model'].calibrated_classifiers_[0]
    pipeline=calibrated.estimator
    z=pipeline[:-1].transform(pd.DataFrame([sample()],columns=FEATURES))[0]
    sigmoid=calibrated.calibrators[0]
    log_odds=-sigmoid.a_*(pipeline.named_steps['model'].intercept_[0]+np.dot(pipeline.named_steps['model'].coef_[0],z))-sigmoid.b_
    expected=1/(1+np.exp(-log_odds))
    assert model.predict_risk(sample())['defect_probability']==pytest.approx(expected)
