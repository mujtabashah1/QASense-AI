import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator_v2.inference import locate_code
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--file',type=Path); parser.add_argument('--top-k',type=int,default=5); args=parser.parse_args()
    code=args.file.read_text(encoding='utf-8') if args.file else 'class Example {\n  int total(int price, int discount) {\n    int result = price + discount;\n    return result;\n  }\n}'
    print(json.dumps(locate_code(code,args.top_k),indent=2))
