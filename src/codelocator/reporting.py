import json
from .config import REPORTS, PROCESSED, MODELS, TRAINING
from .utils import save_json

def load_json(path): return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def generate_report():
    metadata=load_json(PROCESSED/'metadata/metadata.json'); audit=load_json(REPORTS/'metrics/data_audit.json'); baseline=load_json(REPORTS/'metrics/baseline.json'); codebert=load_json(REPORTS/'metrics/codebert_bug_type.json'); localization=load_json(REPORTS/'metrics/localization.json'); status=load_json(REPORTS/'metrics/execution_status.json'); hardware=load_json(REPORTS/'metrics/hardware.json')
    report='# QASense CodeLocator v1 model report\n\n'
    report+='## Objective and execution status\n\nClassify Java repair categories from buggy fragments and, only on verified real context, learn line-level suspiciousness ranking. No universal defect probability is justified by this repair-only dataset.\n\n```json\n'+json.dumps(status,indent=2)+'\n```\n'
    report+='\n## Dataset source and raw schema\n\nPrimary: local ManySStuBs4J sstubsLarge. Alternate source context: sstub_input_no_comments. No other QASense datasets or RiskNet files were modified.\n\n```json\n'+json.dumps({'raw_records':audit['rows'] if audit else None,'schema':audit['column_names'] if audit else None},indent=2)+'\n```\n'
    if metadata:
        report+='\n## Cleaning methodology, counts and duplicates\n\nOnly line endings are normalized. Java case, syntax, punctuation, operators, identifiers, literals and indentation are preserved. Records removed have row-level reasons in tables/dropped_records.csv. Exact standardized duplicate identities include category; conflicting categories for the same fragment are retained as label ambiguity.\n\n```json\n'+json.dumps(metadata['cleaning'],indent=2)+'\n```\n'
        report+='\nRaw duplicates: '+json.dumps({k:v for k,v in audit.items() if 'duplicate' in k})+'. Conflicting-category buggy fragments: '+str(metadata['buggy_fragments_with_conflicting_categories'])+'.\n'
        report+='\n## Bug types and class imbalance\n\n| Category | Raw count | Raw percentage |\n|---|---:|---:|\n'
        for row in audit['bug_type_distribution']: report+=f"| {row['bug_type']} | {row['count']} | {row['percentage']:.2f}% |\n"
        report+='\nRare raw classes: '+str(audit['rare_classes'])+'. Counts by split are saved in tables/split_class_distribution.csv. Throws-exception additions/deletions have identical source fragments in this dataset; the pair validity rule excludes them, not a frequency-based class merge.\n'
        report+='\n## Context availability and localization design\n\n'+(REPORTS/'context_decision.md').read_text().replace('# Source context decision','')+'\n'
        report+='\n## Splits and leakage protections\n\n```json\n'+json.dumps(metadata['splits'],indent=2)+'\n```\n\nRepositories are disjoint. Exact original and repaired fragment hashes, pairs, patches and commits are purged across evaluation boundaries. Purging can change split proportions and makes evaluation conditional on nonduplicated examples. Duplicate sample IDs and label coverage are asserted. Split files are SHA256 checked before each training launch.\n'
    report+='\n## Classical baseline and CodeBERT configuration\n\nBaseline: training-only character TF-IDF (3–5 grams, 50,000 features) and balanced Logistic Regression. CodeBERT is initialized from microsoft/codebert-base, never trained from scratch. No corrected code, diff, project or commit metadata is a model feature.\n\n```json\n'+json.dumps(TRAINING,indent=2)+'\n```\n'
    report+='\n## Frozen and trained layers\n\nLayer indices are zero-based. Embeddings and encoder layers 0–7 are frozen. Only encoder layers 8–11 and each task classifier head (dense and out_proj) are trained. The binary candidate classifier is the localization head. Optimizers include only parameters with requires_grad=True. Non-reentrant gradient checkpointing preserves gradients for the upper layers without an embedding gradient hook.\n\n'
    for task in ['bug_type','localizer']:
        policy=load_json(REPORTS/f'metrics/{task}_layer_policy.json')
        if policy: report+=task+':\n\n```json\n'+json.dumps(policy,indent=2)+'\n```\n\n'
    report+='The full prepared datasets and recorded leakage-safe splits remain unchanged. Training uses up to three epochs, effective batches of 16 with token-bounded microbatches and gradient accumulation, and CUDA FP16 mixed precision. Checkpoint selection uses validation macro F1 for bug types and validation MRR for localization; localization evaluation includes Top-1, Top-3, Top-5 and MRR.\n'
    report+='\n## GPU and hardware\n\n```json\n'+json.dumps(hardware,indent=2)+'\n```\n'
    preflight=load_json(REPORTS/'metrics/preflight.json')
    if preflight: report+='\nPretrained compatibility check (not training/evaluation): '+json.dumps(preflight)+'.\n'
    report+='\n## Classification results and model comparison\n\n| Model | Split | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Top-3 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for name,data in [('TF-IDF',baseline),('CodeBERT',codebert)]:
        if data:
            for split,r in data.items(): report+='| '+name+' | '+split+' | '+' | '.join(f'{r[k]:.4f}' for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','top_3_accuracy'])+' |\n'
    if codebert is None: report+='\nCodeBERT has not completed evaluation. No transformer performance is claimed.\n'
    if baseline: report+='\n![Baseline confusion matrix](figures/baseline_confusion_matrix.png)\n'
    report+='\n## Localization results\n\n'
    if localization:
        report+='```json\n'+json.dumps({k:{m:v for m,v in r.items() if m!='candidate_classification'} for k,r in localization.items()},indent=2)+'\n```\n'
        report+='\nCandidate-line classification metrics are retained in `metrics/localization.json`. '
        test_rank=localization['test']['ranking']; position=localization['test']['position_only_diagnostic']
        if position['mrr']>test_rank['mrr']:
            report+=f"The position-only diagnostic materially outperforms the learned model on test (MRR {position['mrr']:.4f} versus {test_rank['mrr']:.4f}; Top-1 {position['top_1']:.4f} versus {test_rank['top_1']:.4f}). This demonstrates strong target-position bias caused by the verified-context extraction procedure. Learned localization scores are reported, but they must not be interpreted as evidence that the model beats this positional shortcut or generalizes to arbitrary source files.\n"
    else: report+='Top-1, Top-3, Top-5, MRR, mean rank and median rank: unavailable until the localizer is trained and evaluated. Verified context availability does not itself establish model performance.\n'
    if metadata: report+='\nLocalization sample/candidate counts after exact-input leakage purging: '+json.dumps(metadata.get('localization_split_counts'))+'. Excluded source groups: '+str(metadata.get('localization_split_exclusions'))+'. Full-radius windows are required away from true file boundaries, so synthetic crop edges cannot identify the target line. The localizer selects its checkpoint by validation MRR.\n'
    report+='\n## Error analysis and final model\n\n'
    if baseline:
        worst=sorted([(label,r['f1-score'],r['support']) for label,r in baseline['test']['per_class'].items() if isinstance(r,dict) and 'support' in r and label not in ['macro avg','weighted avg']],key=lambda r:r[1])[:5]
        report+='Lowest baseline per-class test F1 values (label, F1, support): '+str(worst)+'.\n'
        mapping=load_json(PROCESSED/'metadata/label_mapping.json'); names=sorted(mapping['label2id'],key=mapping['label2id'].get)
        matrix=baseline['test']['confusion_matrix']
        errors=sorted([(matrix[i][j],names[i],names[j]) for i in range(len(names)) for j in range(len(names)) if i!=j],reverse=True)[:5]
        report+='\nMost frequent baseline confusions (count, actual, predicted): '+str(errors)+'.\n'
    selection=load_json(MODELS/'selection.json')
    report+='\nModel selection uses validation macro F1, never test performance. Selection: '+json.dumps(selection)+'. Per-class scores and full confusion matrices are saved in metrics JSON.\n'
    recovery=load_json(REPORTS/'metrics/localizer_recovery.json')
    if recovery: report+='\n## Interrupted-run recovery\n\nThe saved epoch-2 validation-MRR-selected checkpoint was evaluated without retraining. Epoch 3 did not complete and had no optimizer/scheduler state to resume exactly. Recovery provenance:\n\n```json\n'+json.dumps(recovery,indent=2)+'\n```\n\nSee `LOCALIZER_RECOVERY_AUDIT.md` for the artifact evidence and decision.\n'
    tests=REPORTS/'metrics/test_results.xml'
    if tests.exists():
        import xml.etree.ElementTree as ET
        suites=ET.parse(tests).getroot().findall('testsuite')
        report+='\n## Automated validation\n\n'+json.dumps([{k:s.get(k) for k in ['tests','failures','errors','skipped','time']} for s in suites])+'. Skipped transformer loading tests mean those models were not trained and are not validated.\n'
    report+='\n## Scientifically relevant limitations\n\nTraining examples are Java single-statement repair patterns from ManySStuBs4J, not representative clean versus defective programs. Category confidence is conditional on the repair classification task and is not calibrated defect probability. Multiple labels can describe the same repair. Primary fields are AST fragments; whole-file category inputs are outside that training distribution. Verified-context localization covers a subset selected by available provenance; surrounding lines are weak negatives, not guaranteed bug-free. Candidate scores depend on negative sampling and are suspiciousness scores, not universally calibrated P(defective line). Context selection can induce position artifacts; position-only ranking is evaluated when training completes. CodeBERT truncation can lose relevant context. No claims of universal accuracy for Python, JavaScript, TypeScript, C#, multi-line defects, or exact faulty-line identification are made.\n'
    report+='\n## RiskNet integration\n\nCodeLocator is an independent Python API. A future QASense application can call analyze_code alongside the existing RiskNet interface. RiskNet, frontend, automation and multimodal fusion were not changed.\n'
    report+='\n## Reproduction\n\nFrom C:\\QASense-AI:\n\n```powershell\n.\\.venv-codelocator\\Scripts\\python.exe scripts/preprocess_codelocator.py\n.\\.venv-codelocator\\Scripts\\python.exe scripts/prepare_codebert.py\n.\\.venv-codelocator\\Scripts\\python.exe scripts/train_codelocator.py\n.\\.venv-codelocator\\Scripts\\python.exe scripts/infer_codelocator.py\n.\\.venv-codelocator\\Scripts\\python.exe -m pytest tests/codelocator -q\n```\n\nFor Windows native-library blocks, use scripts/setup_codelocator_wsl.sh in a user Ubuntu WSL2 environment; no repeated package downgrades are required.\n'
    (REPORTS/'CODELOCATOR_MODEL_REPORT.md').write_text(report,encoding='utf-8')
    if baseline:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        mapping=load_json(PROCESSED/'metadata/label_mapping.json'); labels=sorted(mapping['label2id'],key=mapping['label2id'].get)
        for name,data in [('baseline',baseline),('codebert',codebert)]:
            if data:
                fig,ax=plt.subplots(figsize=(12,10)); matrix=np.array(data['test']['confusion_matrix']); image=ax.imshow(matrix,cmap='Blues'); ax.set_xticks(range(len(labels)),labels,rotation=90); ax.set_yticks(range(len(labels)),labels); ax.set_xlabel('Predicted category'); ax.set_ylabel('Actual category'); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(REPORTS/f'figures/{name}_confusion_matrix.png'); plt.close(fig)

if __name__=='__main__': generate_report()
