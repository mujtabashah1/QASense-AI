import math
from pathlib import Path
import joblib
import pandas as pd
from .data_loader import ALIASES
from .utils import ROOT, risk_mapping

class RiskNet:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else ROOT/'models/risknet/risknet_v1.joblib'
        if not path.is_file(): raise FileNotFoundError(f'RiskNet model missing: {path}; run training first')
        # Only load trusted local joblib files: pickle can execute code.
        self.bundle = joblib.load(path)
        if not {'model','features','config','global_importance'} <= self.bundle.keys(): raise ValueError('Invalid model bundle')

    def predict_risk(self, metrics):
        if not isinstance(metrics,dict): raise ValueError('metrics must be a dictionary')
        normalized = {}
        for k,v in metrics.items():
            name = ALIASES.get(str(k).strip().casefold(),k)
            if name in normalized: raise ValueError('Duplicate feature aliases')
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0: raise ValueError(f'Invalid nonnegative finite numeric value for {k}')
            normalized[name] = v
        expected = self.bundle['features']
        if set(normalized) != set(expected): raise ValueError(f'Missing features: {sorted(set(expected)-set(normalized))}; unknown features: {sorted(set(normalized)-set(expected))}')
        p = float(self.bundle['model'].predict_proba(pd.DataFrame([normalized],columns=expected))[0,1])
        level,priority = risk_mapping(p,self.bundle['config'])
        factors = [dict(r,scope='global held-out permutation importance') for r in self.bundle['global_importance'][:5] if r['mean_ap_decrease'] > 0]
        if self.bundle['metadata']['model_type'] == 'logistic':
            classifier = self.bundle['model'].calibrated_classifiers_[0]
            pipeline = classifier.estimator
            z = pipeline[:-1].transform(pd.DataFrame([normalized],columns=expected))[0]
            contributions = -classifier.calibrators[0].a_ * pipeline.named_steps['model'].coef_[0] * z
            factors = sorted([{'feature':name,'log_odds_contribution':float(value),'scope':'local calibrated log odds relative to training feature means'} for name,value in zip(expected,contributions) if value > 0],key=lambda r:r['log_odds_contribution'],reverse=True)[:5]
        return {'defect_probability':p,'prediction':'Defect-Prone' if p >= self.bundle['config']['decision_threshold'] else 'Non-Defect-Prone','risk_level':level,'qa_priority':priority,'top_risk_factors':factors}

def predict_risk(metrics, model_path=None):
    return RiskNet(model_path).predict_risk(metrics)
