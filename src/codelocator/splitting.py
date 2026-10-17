from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .utils import save_json, write_jsonl
from .config import SEED, PROCESSED, REPORTS

def leakage_tokens(record):
    values=[record['buggy_hash'],record['fixed_hash'],record['pair_hash'],record.get('patch_hash')]
    return {'code:'+v for v in values if v} | {'commit:'+record['commit'],'commit:'+record['parent_commit']}

def verify_splits(splits):
    names=list(splits)
    for i,name in enumerate(names):
        records=splits[name]
        if not records: raise ValueError(f'Empty {name} split')
        if any(not r['bug_type'] or not r['buggy_statement'].strip() for r in records): raise ValueError('Missing input or target')
        if len({r['sample_id'] for r in records})!=len(records): raise ValueError('Duplicate sample IDs inside split')
        for other in names[i+1:]:
            for key in ['repository','sample_id','pair_hash','buggy_hash','fixed_hash','commit','parent_commit']:
                if {r[key] for r in records}&{r[key] for r in splits[other]}: raise ValueError(f'Leakage {key}: {name}/{other}')
            if set().union(*(leakage_tokens(r) for r in records)) & set().union(*(leakage_tokens(r) for r in splits[other])): raise ValueError('Cross-field code or commit leakage')
    labels={r['bug_type'] for r in splits['train']}
    if not {r['bug_type'] for r in splits['validation']+splits['test']}<=labels: raise ValueError('Evaluation contains labels absent from training')
    return True

def split_records(records):
    frame=pd.DataFrame({'repository':[r['repository'] for r in records],'label':[r['bug_type'] for r in records]})
    labels=set(frame.label); best=None
    for seed in range(SEED,SEED+80):
        tr,hold=next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=seed).split(frame,groups=frame.repository))
        va,te=next(GroupShuffleSplit(n_splits=1,test_size=.50,random_state=seed).split(frame.iloc[hold],groups=frame.repository.iloc[hold]))
        indices={'train':tr,'validation':hold[va],'test':hold[te]}
        score=sum(abs(len(indices[k])/len(records)-target) for k,target in [('train',.7),('validation',.15),('test',.15)])
        score+=sum(len(labels-set(frame.iloc[idx].label)) for idx in indices.values())
        if best is None or score<best[0]: best=(score,seed,indices)
    _,seed,indices=best
    splits={}; seen=set(); exclusions=[]
    for name in ['train','validation','test']:
        kept=[]
        for idx in sorted(indices[name]):
            r=records[int(idx)]; overlap=leakage_tokens(r)&seen
            if overlap:
                exclusions.append({'sample_id':r['sample_id'],'proposed_split':name,'reason':'code_patch_or_commit_overlaps_earlier_split'}); continue
            kept.append(r)
        splits[name]=kept
        for r in kept: seen.update(leakage_tokens(r))
    verify_splits(splits)
    pd.DataFrame(exclusions,columns=['sample_id','proposed_split','reason']).to_csv(REPORTS/'tables/split_exclusions.csv',index=False)
    manifest=[]; counts=[]
    for name,rows in splits.items():
        write_jsonl(PROCESSED/f'splits/{name}.jsonl',rows)
        manifest.extend({'sample_id':r['sample_id'],'repository':r['repository'],'commit':r['commit'],'bug_type':r['bug_type'],'split':name} for r in rows)
        counts.extend({'split':name,'bug_type':label,'count':count} for label,count in Counter(r['bug_type'] for r in rows).items())
    pd.DataFrame(manifest).to_csv(PROCESSED/'splits/split_manifest.csv',index=False)
    pd.DataFrame(counts).to_csv(REPORTS/'tables/split_class_distribution.csv',index=False)
    info={'strategy':'Repository-exclusive GroupShuffleSplit; deterministic seeds 42..121 search balances proportions and label coverage without looking at model performance. Purge evaluation overlaps against train, then validation; all exclusions documented.','seed':seed,'requested_seed':SEED,'pre_purge_sizes':{k:len(v) for k,v in indices.items()},'sizes':{k:len(v) for k,v in splits.items()},'repositories':{k:len({r['repository'] for r in v}) for k,v in splits.items()},'split_exclusions':len(exclusions),'project_overlap':0,'duplicate_leakage':0,'validated':True}
    save_json(PROCESSED/'metadata/split_metadata.json',info)
    return splits,info
