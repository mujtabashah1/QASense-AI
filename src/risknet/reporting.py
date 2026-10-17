"""Enrich persisted experiment reports without retraining models."""
import json
from .utils import ROOT

def finalize_report():
    report_path = ROOT/'reports/risknet/RISKNET_MODEL_REPORT.md'
    report = report_path.read_text(encoding='utf-8').split('\n<!-- persisted-summary -->')[0]
    report = report.replace('Returned factors explicitly report global importance rather than personal explanations.', 'For logistic models, inference returns positive local calibrated log-odds contributions relative to training feature means; other families return explicitly labeled global importance. Contributions explain the fitted function, not causation.')
    audit = json.loads((ROOT/'reports/risknet/metrics/data_audit.json').read_text())
    nested = json.loads((ROOT/'reports/risknet/metrics/nested_cross_project.json').read_text())
    metadata = json.loads((ROOT/'models/risknet/risknet_v1_metadata.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,5,figsize=(15,3))
    for ax,r in zip(axes,nested):
        matrix = r['confusion_matrix']
        ax.imshow(matrix,cmap='Blues'); ax.set_title(r['project']); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        maximum = max(max(row) for row in matrix)
        for i in range(2):
            for j in range(2): ax.text(j,i,matrix[i][j],ha='center',va='center',color='white' if matrix[i][j] > maximum/2 else 'black')
    fig.tight_layout(); fig.savefig(ROOT/'reports/risknet/figures/confusion_matrices.png'); plt.close(fig)
    report += '\n<!-- persisted-summary -->\n\n## Dataset counts\n\n| Project | Rows | Nondefective | Defective | Defect rate |\n|---|---:|---:|---:|---:|\n'
    for name in ['cm1','jm1','kc1','kc2','pc1']:
        r=audit[name]
        report += f"| {name.upper()} | {r['rows']} | {r['target_distribution']['0']} | {r['target_distribution']['1']} | {r['defect_percentage']:.2f}% |\n"
    report += '\n## Calibrated operating point and probability quality\n\nThe default decision threshold is 0.20. Baselines use 0.50; threshold-free ROC-AUC and AP remain the appropriate ranking comparison. Equal-project nested means: '+', '.join(f'{k}={v:.4f}' for k,v in metadata['evaluation_metrics'].items())+'.\n\n| Project | Raw weighted Brier | Sigmoid Brier |\n|---|---:|---:|\n'
    for r in nested:
        report += f"| {r['project']} | {r['raw_metrics']['brier']:.4f} | {r['brier']:.4f} |\n"
    report += '\nSigmoid calibration improved Brier score on all five held-out projects. The reliability plot still shows overconfidence in higher probability bins, so high-risk probabilities should not be treated as guarantees. Mean recall is below 50%; RiskNet is suitable as an experimental prioritization signal, not a sole QA acceptance gate. Duplicate weighting and project shifts remain material limitations. The 0.20 threshold was specified before baseline results, not optimized using held-out labels.\n'
    ranking = json.loads((ROOT/'reports/risknet/metrics/model_ranking.json').read_text())
    report += '\n## Baseline selection evidence\n\n| Model | Mean cross-project AP | Recall at 0.50 | MCC at 0.50 |\n|---|---:|---:|---:|\n'
    for r in ranking: report += f"| {r['model']} | {r['pr_auc']:.4f} | {r['recall']:.4f} | {r['mcc']:.4f} |\n"
    if metadata['model_type']=='logistic':
        import joblib
        import pandas as pd
        calibrated=joblib.load(ROOT/'models/risknet/risknet_v1.joblib')['model'].calibrated_classifiers_[0]
        coefficients=-calibrated.calibrators[0].a_ * calibrated.estimator.named_steps['model'].coef_[0]
        importance=pd.DataFrame({'feature':metadata['features'],'calibrated_log_odds_per_standard_deviation':coefficients})
        importance['absolute_coefficient']=importance.calibrated_log_odds_per_standard_deviation.abs()
        importance=importance.sort_values('absolute_coefficient',ascending=False)
        importance.to_csv(ROOT/'reports/risknet/metrics/final_model_coefficients.csv',index=False)
        report += '\n## Final artifact coefficients\n\nThese coefficients come from the saved final model. Each is the calibrated log-odds change for a one-standard-deviation feature increase, holding the other metrics fixed. Signs and magnitudes can be unstable with correlated predictors; they are not causal effects. Global held-out permutation importance above is a separate generalization measure.\n\n| Feature | Calibrated log-odds coefficient |\n|---|---:|\n'
        for _,r in importance.head(10).iterrows(): report += f"| {r['feature']} | {r['calibrated_log_odds_per_standard_deviation']:.4f} |\n"
    report_path.write_text(report,encoding='utf-8')

if __name__ == '__main__': finalize_report()
