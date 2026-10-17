import json
from .config import PROCESSED, RAW, REPORTS, TRAINING
from .data_loader import load_primary
from .audit import audit_dataframe
from .cleaning import clean_dataframe
from .context_builder import attach_context, candidate_lines
from .splitting import split_records, verify_splits
from .utils import save_json,write_jsonl,read_jsonl,file_digest,digest

def prepare():
    frame=load_primary(); audit=audit_dataframe(frame)
    print('Actual schema: '+json.dumps(audit['column_names']),flush=True)
    records,cleaning=clean_dataframe(frame)
    print('Conservative cleaning complete: '+json.dumps(cleaning),flush=True)
    context=attach_context(records)
    print('Verified context: '+json.dumps(context),flush=True)
    write_jsonl(PROCESSED/'clean/codelocator_clean.jsonl',records)
    write_jsonl(PROCESSED/'clean/bug_type_dataset.jsonl',({'sample_id':r['sample_id'],'repository':r['repository'],'text':r['buggy_statement'],'label':r['bug_type']} for r in records))
    write_jsonl(PROCESSED/'clean/localization_dataset.jsonl',(c for r in records for c in candidate_lines(r)))
    splits,split_info=split_records(records)
    labels=sorted({r['bug_type'] for r in splits['train']})
    save_json(PROCESSED/'metadata/label_mapping.json',{'label2id':{v:i for i,v in enumerate(labels)},'id2label':{str(i):v for i,v in enumerate(labels)}})
    localizer_seen=set(); localizer_exclusions=[]; localizer_counts={}
    for name,rows in splits.items():
        candidates=[]; hashes=set(); sample_count=0
        for r in rows:
            group=candidate_lines(r,TRAINING['negative_lines_per_sample'] if name=='train' else None)
            if not group: continue
            keys={digest(c['candidate_line']+'\n\n'+c['context']) for c in group}
            if keys & localizer_seen:
                localizer_exclusions.append({'sample_id':r['sample_id'],'split':name,'reason':'exact_localizer_input_overlaps_earlier_split'}); continue
            if sum(c['label'] for c in group)!=1: raise ValueError('Invalid localization labels')
            candidates.extend(group); hashes.update(keys); sample_count+=1
        write_jsonl(PROCESSED/f'splits/{name}_localization.jsonl',candidates)
        localizer_seen.update(hashes)
        localizer_counts[name]={'source_samples':sample_count,'candidate_lines':len(candidates)}
    import pandas as pd
    pd.DataFrame(localizer_exclusions,columns=['sample_id','split','reason']).to_csv(REPORTS/'tables/localization_split_exclusions.csv',index=False)
    files=[RAW/'sstubsLarge/train-00000-of-00001.parquet']+sorted((RAW/'sstub_input_no_comments').glob('*.parquet'))
    metadata={'version':'1.0.0','raw_rows':len(frame),'cleaning':cleaning,'context':context,'splits':split_info,'bug_classes':len(labels),'clean_repositories':len({r['repository'] for r in records}),'raw_sha256':{str(p.relative_to(RAW)):file_digest(p) for p in files},'processed_sha256':{str(p.relative_to(PROCESSED)):file_digest(p) for p in (PROCESSED/'splits').glob('*.jsonl')},'training_config':TRAINING,'data_quality_gate_passed':True,'predictive_fields':['buggy_statement'],'localization_predictive_fields':['candidate_line','context'],'excluded_predictive_fields':['fixed_statement','fixed_code','patch','buggy_line_number','bug_type','repository','commit']}
    # Document conflicting labels rather than hiding task ambiguity.
    groups={}
    for r in records: groups.setdefault(r['buggy_hash'],set()).add(r['bug_type'])
    metadata['buggy_fragments_with_conflicting_categories']=sum(len(v)>1 for v in groups.values())
    metadata['localization_split_counts']=localizer_counts
    metadata['localization_split_exclusions']=len(localizer_exclusions)
    save_json(PROCESSED/'metadata/metadata.json',metadata)
    save_json(REPORTS/'metrics/preprocessing.json',metadata)
    (REPORTS/'context_decision.md').write_text('# Source context decision\n\n'+json.dumps(context,indent=2)+'\n\nPrimary source fields are fragments. Alternate prefix + original bug + suffix reconstruct real pre-fix files. Only exact joins with verified line and character offset are accepted. Candidate inputs center every candidate using the same context rule; neither original bug index nor a patch is exposed. Other context lines are weak negatives: they are not independently proven bug free. Ranking evaluation is conditional on verified Java source context and does not establish full-repository localization.\n',encoding='utf-8')
    validate_processed()
    print('DATA AUDIT COMPLETE\n'+json.dumps({'raw_samples':len(frame),'clean_samples':len(records),'dropped':cleaning['dropped'],'bug_classes':len(labels),'repositories':metadata['clean_repositories'],**split_info,'context_available':context['genuine_context_available'],'localization_supported':context['localization_scientifically_supported']},indent=2),flush=True)
    return metadata

def validate_processed():
    path=PROCESSED/'metadata/metadata.json'
    if not path.exists(): raise ValueError('Run preprocessing before training')
    metadata=json.loads(path.read_text(encoding='utf-8'))
    if not metadata.get('data_quality_gate_passed'): raise ValueError('Data quality gate not passed')
    clean_path=PROCESSED/'clean/codelocator_clean.jsonl'
    if not clean_path.is_file() or not clean_path.stat().st_size: raise ValueError('Clean dataset missing or empty')
    for name,sha in metadata['processed_sha256'].items():
        if file_digest(PROCESSED/name)!=sha: raise ValueError('Processed split changed; rerun preprocessing')
    splits={n:read_jsonl(PROCESSED/f'splits/{n}.jsonl') for n in ['train','validation','test']}
    verify_splits(splits)
    mapping=json.loads((PROCESSED/'metadata/label_mapping.json').read_text())
    if set(mapping['label2id'])!={r['bug_type'] for r in splits['train']}: raise ValueError('Incorrect label mapping')
    expected={label:i for i,label in enumerate(sorted(mapping['label2id']))}
    if mapping['label2id']!=expected or mapping['id2label']!={str(i):label for label,i in expected.items()}: raise ValueError('Label IDs must be deterministic contiguous inverse mappings')
    if any(len(v)<20 for v in splits.values()): raise ValueError('Split too small')
    if not (REPORTS/'context_decision.md').is_file() or not (REPORTS/'tables/split_class_distribution.csv').is_file(): raise ValueError('Missing context decision or distribution')
    return splits,metadata,mapping
