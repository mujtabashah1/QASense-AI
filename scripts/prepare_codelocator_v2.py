import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator_v2.preprocessing import prepare_v2

if __name__=='__main__':
    gate=prepare_v2()
    print(json.dumps(gate,indent=2),flush=True)
    print('PRE-TRAINING STOP: inspect reports/codelocator_v2/metrics/pretraining_gate.json before any v2 GPU training.',flush=True)
