"""Run audit, grouped baselines, nested tuning/calibration and final refit."""
import platform
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from .data_loader import audit_and_clean, FEATURES
from .preprocessing import make_pipeline
from .evaluate import evaluate
from .utils import ROOT, CONFIG, save_json

NAMES = ['logistic', 'random_forest', 'xgboost']

def grouped_cv(frame, groups, n=3):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42).split(frame[FEATURES], frame.defect, groups))

def tuned_model(name, frame, groups):
    params = {'logistic': {'model__C': [.01,.1,1,10]}, 'random_forest': {'model__max_depth': [6,12,None], 'model__min_samples_leaf': [2,5,10]}, 'xgboost': {'model__max_depth': [2,3,5], 'model__learning_rate': [.03,.07,.12]}}[name]
    search = RandomizedSearchCV(make_pipeline(name,frame.defect), params, n_iter=4, scoring='average_precision', cv=grouped_cv(frame,groups), random_state=42, n_jobs=1)
    search.fit(frame[FEATURES],frame.defect)
    return search.best_estimator_, search.best_params_, float(search.best_score_)

def table(records):
    keys = ['protocol','project','model','precision','recall','f1','roc_auc','pr_auc','mcc','brier']
    out = '| ' + ' | '.join(keys) + ' |\n|' + '|'.join(['---']*len(keys)) + '|\n'
    for r in records:
        out += '| ' + ' | '.join(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]) for k in keys) + ' |\n'
    return out

def main():
    frame, groups, audit = audit_and_clean()
    records, folds = [], []
    for project in frame.project.unique():
        idx = np.flatnonzero(frame.project.eq(project))
        tr, te = next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(frame.iloc[idx][FEATURES],frame.iloc[idx].defect,groups.iloc[idx]))
        folds.append(('within',project,idx[tr],idx[te],0))
        test = np.flatnonzero(frame.project.eq(project))
        candidate = ~frame.project.eq(project)
        overlap = groups.isin(set(groups.iloc[test]))
        train = np.flatnonzero(candidate & ~overlap)
        folds.append(('cross',project,train,test,int((candidate & overlap).sum())))
    manifests = []
    for protocol,project,tr,te,removed in folds:
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])
        manifests.append({'protocol':protocol,'project':project,'train_indices':tr.tolist(),'test_indices':te.tolist(),'purged_training_rows':removed})
        for name in NAMES:
            model = make_pipeline(name,frame.iloc[tr].defect)
            model.fit(frame.iloc[tr][FEATURES],frame.iloc[tr].defect)
            p = model.predict_proba(frame.iloc[te][FEATURES])[:,1]
            records.append(dict(protocol=protocol,project=project,model=name,train_rows=len(tr),test_rows=len(te),purged_training_rows=removed,**evaluate(frame.iloc[te].defect,p)))
        print(f'Completed baselines: {protocol} {project}', flush=True)
    baseline = pd.DataFrame(records)
    save_json(ROOT/'reports/risknet/metrics/baselines.json',records)
    save_json(ROOT/'reports/risknet/metrics/split_manifest.json',manifests)
    cross = baseline[baseline.protocol.eq('cross')]
    # Select algorithm using equal-project mean AP, then recall and MCC.
    ranking = cross.groupby('model')[['pr_auc','recall','mcc']].mean().sort_values(['pr_auc','recall','mcc'],ascending=False)
    winner = ranking.index[0]
    print(f'Selected baseline family: {winner}',flush=True)
    nested, importance_vectors, calibration_y, calibration_p = [], [], [], []
    for protocol, project, tr, te, removed in folds:
        if protocol != 'cross':
            continue
        training = frame.iloc[tr]
        model, parameters, cv_ap = tuned_model(winner,training,groups.iloc[tr])
        splits = grouped_cv(training,groups.iloc[tr])
        calibrated = CalibratedClassifierCV(model, method='sigmoid',cv=splits,ensemble=False)
        calibrated.fit(training[FEATURES],training.defect)
        p_raw = model.predict_proba(frame.iloc[te][FEATURES])[:,1]
        p = calibrated.predict_proba(frame.iloc[te][FEATURES])[:,1]
        nested.append(dict(protocol='nested_cross_sigmoid',project=project,model=winner,parameters=parameters,inner_cv_ap=cv_ap,raw_metrics=evaluate(frame.iloc[te].defect,p_raw),threshold_comparison={str(t):evaluate(frame.iloc[te].defect,p,t) for t in [.1,.2,.3,.5]},**evaluate(frame.iloc[te].defect,p,CONFIG['decision_threshold'])))
        imp = permutation_importance(calibrated,frame.iloc[te][FEATURES],frame.iloc[te].defect,scoring='average_precision',n_repeats=3,random_state=42,n_jobs=2)
        importance_vectors.append(imp.importances_mean)
        calibration_y.extend(frame.iloc[te].defect.tolist())
        calibration_p.extend(p.tolist())
        print(f'Completed nested tuning and calibration: {project}',flush=True)
    final_model, best_params, cv_ap = tuned_model(winner,frame,groups)
    final = CalibratedClassifierCV(final_model,method='sigmoid',cv=grouped_cv(frame,groups),ensemble=False)
    final.fit(frame[FEATURES],frame.defect)
    importance = pd.DataFrame({'feature':FEATURES,'mean_ap_decrease':np.mean(importance_vectors,axis=0),'project_std':np.std(importance_vectors,axis=0)}).sort_values('mean_ap_decrease',ascending=False)
    importance.to_csv(ROOT/'reports/risknet/metrics/feature_importance.csv',index=False)
    aggregate = {k: float(np.mean([r[k] for r in nested])) for k in ['precision','recall','f1','roc_auc','pr_auc','mcc','brier']}
    metadata = {'version':'1.0.0','model_type':winner,'training_date':datetime.now(timezone.utc).isoformat(),'features':FEATURES,'projects':frame.project.unique().tolist(),'training_rows':len(frame),'configuration':CONFIG,'parameters':best_params,'inner_cv_ap':cv_ap,'evaluation_metrics':aggregate,'evaluation_note':'Equal-project means of nested LOPO estimates; algorithm family selected using these projects, so not an untouched external test.','calibration':'sigmoid on grouped out-of-fold training predictions','python':platform.python_version(),'raw_sha256':{p:audit[p]['sha256'] for p in ['cm1','jm1','kc1','kc2','pc1']}}
    bundle = {'model':final,'features':FEATURES,'config':CONFIG,'metadata':metadata,'global_importance':importance.to_dict(orient='records')}
    joblib.dump(bundle,ROOT/'models/risknet/risknet_v1.joblib')
    save_json(ROOT/'models/risknet/risknet_v1_metadata.json',metadata)
    save_json(ROOT/'reports/risknet/metrics/baselines.json',records)
    save_json(ROOT/'reports/risknet/metrics/nested_cross_project.json',nested)
    save_json(ROOT/'reports/risknet/metrics/split_manifest.json',manifests)
    save_json(ROOT/'reports/risknet/metrics/model_ranking.json',ranking.reset_index().to_dict(orient='records'))
    pd.DataFrame(records).to_csv(ROOT/'reports/risknet/metrics/baselines.csv',index=False)
    plt.figure(figsize=(9,6)); plt.barh(importance.feature.iloc[:10][::-1],importance.mean_ap_decrease.iloc[:10][::-1]); plt.xlabel('Held-out AP decrease (permutation)'); plt.tight_layout(); plt.savefig(ROOT/'reports/risknet/figures/feature_importance.png'); plt.close()
    frac,mean = calibration_curve(calibration_y,calibration_p,n_bins=10)
    plt.plot(mean,frac,'o-',label='LOPO sigmoid'); plt.plot([0,1],[0,1],'--'); plt.xlabel('Predicted probability'); plt.ylabel('Observed defect fraction'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'reports/risknet/figures/calibration.png'); plt.close()
    report = '# QASense RiskNet v1 model report\n\n'
    report += '## Objective and dataset overview\n\nPredict defective modules from 21 static metrics. Projects are metadata, never predictors. All '+str(len(frame))+' source rows are retained.\n\n'
    report += '## Data quality and class imbalance\n\nSee [data audit](data_audit.md) for per-column statistics, invalid tokens, outliers, correlations, duplicates and class counts. Combined findings: '+str(audit['combined'])+'\n\n'
    report += '## Preprocessing and leakage precautions\n\nExplicit case-insensitive aliases are collision checked, targets validated, numeric invalid tokens/infinities converted to missing. Training-only median imputation and scaling live inside pipelines. No SMOTE. Exact feature vectors are SHA256 grouped, including missing values. Conflicting labels are preserved. Within-project splits use stratified group folds; LOPO purges training matches to test features. Split indices and purge counts are saved. StratifiedGroupKFold is used instead of StratifiedKFold because duplicate groups must stay together. Class weights use training labels; XGBoost scale_pos_weight is recomputed from labels on every fit, including inner folds.\n\n'
    report += '## Models tested and within/cross-project results\n\nLogistic regression, random forest and XGBoost use class weighting. Baseline decision threshold is 0.5. AP is average precision, not trapezoidal PR area.\n\n'+table(records)
    report += '\n## Final model selection and tuning\n\nSelected '+winner+' by equal-project mean cross-project AP, then recall and MCC. Four randomized candidates use three stratified group folds. Nested LOPO repeats tuning using only outer training projects; final artifact refits on all rows. Family selection used the benchmark projects, so reported results retain selection bias. No independent external validation exists.\n\n'+table(nested)
    report += '\n## Probability calibration\n\nSigmoid calibration uses group-separated out-of-fold training predictions, never outer test labels. Weighted raw scores need calibration; raw and calibrated Brier/log loss are saved per project. Isotonic was omitted because small-project positives make it unstable. Calibration transfer to new projects is uncertain.\n\n![Calibration](figures/calibration.png)\n\n## Confusion matrices\n\nRows are actual classes, columns predicted classes, ordered 0/1.\n\n![Matrices](figures/confusion_matrices.png)\n\n'
    report += '## Feature importance\n\nPermutation AP decrease averaged equally across held-out projects; correlations can dilute importance. This is global importance, not individual causal attribution.\n\n![Importance](figures/feature_importance.png)\n\n'+importance.to_string(index=False)+'\n\n'
    report += '## Known limitations and QASense integration\n\nNASA labels are historical, duplicates can carry contradictory labels, metrics are correlated, and these results do not establish performance on modern repositories. A single deterministic within-project split does not measure split sensitivity; cross-project standard deviations describe domain instability. Final calibrated probabilities may still drift. Risk thresholds 0.3/0.6/0.8 and priorities P4/P3/P2/P1 are configurable product defaults, not validated scientific cutoffs. Defect classification uses a separate configurable 0.2 threshold, selected before training to prioritize recall because missed defects are expensive. It is a product operating assumption, not an optimized threshold. Metrics at 0.1/0.2/0.3/0.5 are saved for comparison; test labels were not used to choose the threshold. Inference validates all 21 metrics and returns probability, classification and priority. For logistic models, inference returns positive local calibrated log-odds contributions relative to training feature means; other families return explicitly labeled global importance. Contributions explain the fitted function, not causation. Other QASense modules were not changed.\n\n'
    report += '## Reproduction\n\n```powershell\npython -m venv .venv-risknet\n.\\.venv-risknet\\Scripts\\python.exe -m pip install -r requirements-risknet.txt\n.\\.venv-risknet\\Scripts\\python.exe scripts/train_risknet.py\n.\\.venv-risknet\\Scripts\\python.exe -m pytest tests/risknet -q\n.\\.venv-risknet\\Scripts\\python.exe scripts/infer_risknet.py\n```\n'
    report += '\n## Cross-project stability\n\n'+cross.groupby('model')[['recall','f1','pr_auc','mcc']].agg(['mean','std']).to_string()+'\n'
    (ROOT/'reports/risknet/RISKNET_MODEL_REPORT.md').write_text(report,encoding='utf-8')
    from .reporting import finalize_report
    finalize_report()
    print(metadata,flush=True)

if __name__ == '__main__': main()



