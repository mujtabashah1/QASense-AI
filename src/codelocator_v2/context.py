import hashlib
import random
import re
from .config import MAX_REGION_LINES,SEED,TRAIN_NEGATIVES

CONTROL=re.compile(r'\b(if|for|while|switch|catch|synchronized|class|interface|enum|record)\s*\(')
IDENT=re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*')

def stable_int(value): return int(hashlib.sha256(value.encode()).hexdigest()[:16],16)

def _brace_pairs(lines):
    stack=[]; pairs=[]; block_comment=False
    for line_no,line in enumerate(lines):
        clean=''; i=0; quote=None
        while i<len(line):
            if block_comment:
                end=line.find('*/',i)
                if end<0: break
                block_comment=False; i=end+2; continue
            if quote:
                if line[i]=='\\': i+=2; continue
                if line[i]==quote: quote=None
                i+=1; continue
            if line.startswith('//',i): break
            if line.startswith('/*',i): block_comment=True; i+=2; continue
            if line[i] in "'\"": quote=line[i]; i+=1; continue
            clean+=line[i]; i+=1
        for char in clean:
            if char=='{': stack.append(line_no)
            elif char=='}' and stack:
                start=stack.pop(); pairs.append((start,line_no))
    return pairs

def natural_region(lines,bug_index,sample_id,max_lines=MAX_REGION_LINES):
    """Choose a method/block when possible, then deterministically debias long crops."""
    containing=sorted((p for p in _brace_pairs(lines) if p[0]<=bug_index<=p[1]),key=lambda p:p[1]-p[0])
    chosen=None; kind='source_crop'
    for start,end in containing:
        signature=' '.join(lines[max(0,start-3):start+1])
        if '(' in signature and ')' in signature and not CONTROL.search(signature):
            chosen=(start,end); kind='method'; break
    if chosen is None and containing:
        chosen=next((p for p in containing if p[1]-p[0]+1>=5),containing[-1]); kind='block'
    if chosen is None: chosen=(0,len(lines)-1)
    lower=max(chosen[0],bug_index-max_lines+1); upper=min(bug_index,chosen[1]-max_lines+1)
    if chosen[1]-chosen[0]+1<=max_lines:
        start,end=chosen
    elif lower<=upper:
        target_bucket=stable_int(sample_id+':position')%5
        desired=target_bucket/4 if target_bucket else 0.0
        start=round(bug_index-desired*(max_lines-1)); start=min(upper,max(lower,start)); end=start+max_lines-1
        kind+='_debiased_crop'
    else:
        start=max(chosen[0],min(bug_index,chosen[1])); end=min(chosen[1],start+max_lines-1)
    return {'lines':lines[start:end+1],'start':start,'end':end,'bug_index':bug_index-start,'kind':kind}

def eligible(line):
    stripped=line.strip()
    return bool(stripped and stripped not in {'{','}','};'} and not stripped.startswith(('//','/*','*','*/')))

def _hardness(candidate,bug_line):
    a=set(IDENT.findall(candidate)); b=set(IDENT.findall(bug_line))
    lexical=len(a&b)*4
    structural=sum(token in candidate and token in bug_line for token in ['if','return','=','(','.','&&','||','+','-','==','!='])
    return lexical+structural

def candidates(record,region,split):
    absolute_bug=record['buggy_line_number']-1
    rows=[]
    for offset,line in enumerate(region['lines']):
        absolute=region['start']+offset
        if not eligible(line) and absolute!=absolute_bug: continue
        rows.append({'sample_id':record['sample_id'],'repository':record['repository'],'line_number':absolute+1,'candidate_line':line,'context':'\n'.join(region['lines']),'label':int(absolute==absolute_bug),'region_start_line':region['start']+1,'region_end_line':region['end']+1,'region_kind':region['kind']})
    if sum(row['label'] for row in rows)!=1 or len(rows)<5: return []
    if split=='train':
        positive=[r for r in rows if r['label']]
        negative=sorted((r for r in rows if not r['label']),key=lambda r:(-_hardness(r['candidate_line'],positive[0]['candidate_line']),stable_int(record['sample_id']+':'+str(r['line_number']))))
        rows=positive+negative[:TRAIN_NEGATIVES]
    random.Random(SEED+stable_int(record['sample_id']+':order')).shuffle(rows)
    return rows

def model_text(row): return row['candidate_line']+'\n\n'+row['context']
