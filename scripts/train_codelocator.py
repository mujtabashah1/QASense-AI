import argparse
import json
import sys
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator.config import REPORTS,MODELS
from codelocator.utils import save_json
from codelocator.preprocessing import validate_processed
from codelocator.reporting import generate_report

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--baseline-only',action='store_true'); parser.add_argument('--reuse-baseline',action='store_true'); parser.add_argument('--evaluate-localizer-only',action='store_true',help='Evaluate the existing validation-MRR-selected localizer; never trains'); args=parser.parse_args()
    status={'status':'running','stage':'data_quality_validation','completed':[]}
    path=REPORTS/'metrics/execution_status.json'
    try:
        splits,metadata,mapping=validate_processed(); status['completed'].append('processed_data_validation'); save_json(path,status)
        if args.evaluate_localizer_only:
            status.update(stage='line_localizer_evaluation_recovery',recovery={'training_performed':False,'checkpoint':'saved best after epoch 2'}); save_json(path,status)
            from codelocator.train_localizer import evaluate_existing_localizer
            evaluate_existing_localizer(metadata)
            status['completed'].extend(['baseline','codebert_bug_type','line_localizer'])
            status.update(status='trained',stage='report'); save_json(path,status); generate_report(); return
        status['stage']='baseline'
        from codelocator.baseline import train_baseline,training_signature
        if args.reuse_baseline and (REPORTS/'metrics/baseline.json').exists():
            cached=json.loads((MODELS/'baseline/metadata.json').read_text())
            if cached.get('training_signature')!=training_signature(splits['train']): raise ValueError('Baseline training inputs changed; rerun without --reuse-baseline')
            if cached.get('feature_policy',{}).get('lowercase') is not False: raise ValueError('Baseline must preserve Java identifier case; rerun without --reuse-baseline')
            baseline=json.loads((REPORTS/'metrics/baseline.json').read_text())
        else: baseline=train_baseline(splits,mapping)
        status['completed'].append('baseline'); save_json(path,status)
        selection={'bug_type_backend':'baseline','criterion':'validation macro F1','baseline_validation_macro_f1':baseline['validation']['macro_f1'],'transformer_evaluation_complete':False}
        save_json(MODELS/'selection.json',selection)
        if args.baseline_only:
            status.update(status='baseline_only',stage='baseline_saved'); save_json(path,status); generate_report(); return
        status['stage']='hardware_validation'; save_json(path,status)
        from codelocator.train_bug_type import hardware_probe,train_bug_type
        hardware_probe(); status['stage']='codebert_bug_type'; save_json(path,status)
        codebert=train_bug_type(splits,mapping); status['completed'].append('codebert_bug_type'); save_json(path,status)
        selection.update(transformer_evaluation_complete=True,codebert_validation_macro_f1=codebert['validation']['macro_f1'])
        if codebert['validation']['macro_f1']>baseline['validation']['macro_f1']: selection['bug_type_backend']='codebert'
        save_json(MODELS/'selection.json',selection)
        status['stage']='line_localizer'; save_json(path,status)
        from codelocator.train_localizer import train_localizer
        localized=train_localizer(metadata)
        status['completed'].append('line_localizer' if localized else 'localizer_unsupported_documented')
        status.update(status='trained',stage='report'); save_json(path,status); generate_report()
    except Exception as e:
        status.update(status='blocked_or_failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()); save_json(path,status); generate_report(); raise

if __name__=='__main__': main()
