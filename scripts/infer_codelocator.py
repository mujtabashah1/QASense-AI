import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator.inference import analyze_code
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--file',type=Path); parser.add_argument('--backend',choices=['auto','baseline','codebert'],default='auto'); args=parser.parse_args()
    code=args.file.read_text(encoding='utf-8') if args.file else 'if (quantity >= 0 && price != null) {\n    total = price + discount;\n}'
    print(json.dumps(analyze_code(code,args.backend),indent=2))
