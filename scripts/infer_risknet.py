import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from risknet.data_loader import FEATURES
from risknet.inference import predict_risk
if __name__ == '__main__':
    metrics = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv)>1 else pd.read_csv(Path(__file__).resolve().parents[1]/'datasets/processed/risknet/cm1_clean.csv').iloc[1][FEATURES].astype(float).to_dict()
    print(json.dumps(predict_risk(metrics),indent=2))
