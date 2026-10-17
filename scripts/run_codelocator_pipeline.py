import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from codelocator.utils import save_json

def run(args): subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

if __name__=='__main__':
    steps=[('environment',['-c','import pandas,numpy,sklearn; print("Data environment validated")']),('audit_clean_split_validate',['scripts/preprocess_codelocator.py']),('prepare_codebert',['scripts/prepare_codebert.py']),('train_evaluate_save',['scripts/train_codelocator.py']),('tests',['-m','pytest','tests/codelocator','-q','--junitxml=reports/codelocator/metrics/test_results.xml']),('inference',['scripts/infer_codelocator.py']),('report',['-c','import sys; sys.path.insert(0,"src"); from codelocator.reporting import generate_report; generate_report()'])]
    path=ROOT/'reports/codelocator/metrics/pipeline_execution.json'
    status={'status':'running','completed':[]}
    try:
        for stage,args in steps:
            status['stage']=stage; save_json(path,status); run(args); status['completed'].append(stage); save_json(path,status)
        status['status']='complete'; save_json(path,status)
    except Exception as error:
        status.update(status='failed',error=str(error)); save_json(path,status); raise
