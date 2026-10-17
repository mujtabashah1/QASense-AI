import hashlib
import numpy as np
import pandas as pd
from .utils import ROOT, save_json

FEATURES = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']
ALIASES = {c.casefold(): c for c in FEATURES}
ALIASES.update({'locodeandcomment': 'locCodeAndComment', 'defects': 'defect', 'problems': 'defect', 'defect': 'defect'})

def normalize_schema(frame):
    names = [ALIASES.get(str(c).strip().casefold(), str(c).strip()) for c in frame.columns]
    if len(names) != len(set(names)):
        raise ValueError('Column aliases collide')
    frame = frame.copy()
    frame.columns = names
    missing = set(FEATURES + ['defect']) - set(names)
    extra = set(names) - set(FEATURES + ['defect'])
    if missing or extra:
        raise ValueError(f'Schema mismatch: missing={missing}, extra={extra}')
    labels = frame.defect.astype(str).str.strip().str.casefold().map({'false': 0, 'true': 1, 'no': 0, 'yes': 1, '0': 0, '1': 1})
    if labels.isna().any():
        raise ValueError('Unknown defect labels')
    frame['defect'] = labels.astype(int)
    for c in FEATURES:
        frame[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return frame[FEATURES + ['defect']]

def feature_groups(frame):
    # Exact float values with a stable missing-value marker; no rounding.
    keys = frame[FEATURES].apply(lambda row: '|'.join('NA' if pd.isna(v) else float(v).hex() for v in row), axis=1)
    return keys.map(lambda s: hashlib.sha256(s.encode()).hexdigest())

def audit_and_clean():
    frames, audit = [], {}
    for name in ['cm1', 'jm1', 'kc1', 'kc2', 'pc1']:
        path = ROOT / 'datasets/NASA-promise-dataset-repository-main' / (name + '.csv')
        raw = pd.read_csv(path)
        clean = normalize_schema(raw)
        numeric = clean[FEATURES]
        corr = numeric.corr()
        invalid = {c: int((raw[old].notna() & pd.to_numeric(raw[old], errors='coerce').isna()).sum()) for old, c in zip(raw.columns, [ALIASES.get(c.casefold(), c) for c in raw.columns]) if c in FEATURES}
        audit[name] = {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(raw), 'columns': len(raw.columns), 'column_names': list(raw.columns), 'dtypes': raw.dtypes.astype(str).to_dict(), 'missing_values': raw.isna().sum().to_dict(), 'duplicate_rows': int(raw.duplicated().sum()), 'target_distribution': clean.defect.value_counts().to_dict(), 'defect_percentage': float(clean.defect.mean()*100), 'numeric_statistics': numeric.describe().replace([np.inf, -np.inf], np.nan).where(lambda x: x.notna(), None).to_dict(), 'constant_or_near_constant': [c for c in FEATURES if numeric[c].value_counts(normalize=True, dropna=False).max() >= .99], 'high_correlations': [{'a': a, 'b': b, 'correlation': float(corr.loc[a,b])} for i,a in enumerate(FEATURES) for b in FEATURES[i+1:] if abs(corr.loc[a,b]) >= .95], 'invalid_numeric_values': invalid, 'infinite_values': {c: int(np.isinf(pd.to_numeric(raw[old], errors='coerce')).sum()) for old,c in zip(raw.columns, [ALIASES.get(c.casefold(), c) for c in raw.columns]) if c in FEATURES}, 'negative_values': (numeric < 0).sum().to_dict(), 'extreme_outliers_3_iqr': {c: int(((numeric[c] < numeric[c].quantile(.25)-3*(numeric[c].quantile(.75)-numeric[c].quantile(.25))) | (numeric[c] > numeric[c].quantile(.75)+3*(numeric[c].quantile(.75)-numeric[c].quantile(.25)))).sum()) for c in FEATURES}}
        clean['project'] = name.upper()
        clean.to_csv(ROOT / f'datasets/processed/risknet/{name}_clean.csv', index=False)
        frames.append(clean)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(ROOT / 'datasets/processed/risknet/nasa_combined_clean.csv', index=False)
    groups = feature_groups(combined)
    summary = combined.assign(group=groups).groupby('group').agg(rows=('defect','size'), labels=('defect','nunique'), projects=('project','nunique'))
    duplicate_details = combined[['project','defect']].assign(row_index=combined.index, feature_group=groups)
    duplicate_details[groups.isin(summary.index[summary.rows > 1])].to_csv(ROOT/'reports/risknet/metrics/duplicate_feature_rows.csv',index=False)
    audit['combined'] = {'rows': len(combined), 'unique_feature_groups': len(summary), 'duplicate_feature_groups': int((summary.rows > 1).sum()), 'duplicate_feature_rows_beyond_first': int((summary.rows-1).sum()), 'conflicting_label_groups': int((summary.labels > 1).sum()), 'cross_project_duplicate_groups': int((summary.projects > 1).sum()), 'cross_project_rows': int(summary.loc[summary.projects > 1,'rows'].sum())}
    save_json(ROOT/'reports/risknet/metrics/data_audit.json', audit)
    text = '# NASA dataset audit\n\nRaw files are preserved. Invalid numeric tokens and infinities become missing values; median imputation is fit inside training folds. Outliers, negative values, duplicate rows and conflicting labels are retained and documented, not silently deleted. Near constant means at least 99% one value; high correlation means |r| >= 0.95; extreme means outside 3 IQR.\n\n'
    for name, result in audit.items():
        text += f'## {name}\n\n```json\n' + __import__('json').dumps(result, indent=2, default=str) + '\n```\n\n'
    (ROOT/'reports/risknet/data_audit.md').write_text(text, encoding='utf-8')
    return combined, groups, audit
