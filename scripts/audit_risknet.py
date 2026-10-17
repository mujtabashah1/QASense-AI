import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from risknet.data_loader import audit_and_clean
if __name__ == '__main__':
    _,_,audit = audit_and_clean()
    print(audit['combined'])
