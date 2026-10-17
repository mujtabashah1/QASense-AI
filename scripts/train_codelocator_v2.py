import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator_v2.train import train_v2
if __name__=='__main__': train_v2()
