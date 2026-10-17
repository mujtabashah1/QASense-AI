import hashlib
import json
from pathlib import Path

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def write_jsonl(path,records):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]

def repository_key(value):
    from urllib.parse import urlparse
    value=str(value).strip().removesuffix('.git')
    if '://' in value:
        parts=urlparse(value).path.strip('/').split('/')
        return '/'.join(parts[-2:]).casefold()
    return value.replace('.', '/',1).casefold()
