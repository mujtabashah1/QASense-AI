import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from codelocator.cleaning import normalize_text_field
from codelocator.context_builder import alternate_key
from codelocator.utils import digest,file_digest,read_jsonl,save_json,write_jsonl
from .baselines import position_baseline,random_baseline
from .config import RAW,V1_PROCESSED,PROCESSED,REPORTS,POSITION_MRR_LIMIT,MIDDLE_BUCKET_LIMIT,SINGLE_BUCKET_LIMIT
from .context import candidates,model_text,natural_region

SPLITS=['train','validation','test']

def _allowed_ids():
    return {split:{row['sample_id'] for row in read_jsonl(V1_PROCESSED/f'splits/{split}_localization.jsonl')} for split in SPLITS}

def _source_records(allowed):
    records={}
    for split in SPLITS:
        with (V1_PROCESSED/f'splits/{split}.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                row=json.loads(line)
                if row['sample_id'] in allowed[split]:
                    row={'sample_id':row['sample_id'],'repository':row['repository'],'parent_commit':row['parent_commit'],'buggy_statement':row['buggy_statement'],'fixed_statement':row['fixed_statement'],'bug_type':row['bug_type'],'buggy_line_number':row['buggy_line_number'],'bug_node_start':row['bug_node_start'],'split':split}
                    records[alternate_key(row['repository'],row['parent_commit'],row['buggy_statement'],row['fixed_statement'],row['bug_type'])]=row
    return records

def _position_summary(groups):
    values=[]
    for rows in groups.values():
        positive=next(row for row in rows if row['label'])
        values.append((positive['line_number']-positive['region_start_line'])/max(1,positive['region_end_line']-positive['region_start_line']))
    buckets=[0]*5
    for value in values: buckets[min(4,int(value*5))]+=1
    n=len(values)
    exact=sum(abs(value-.5)<1e-12 for value in values)/n
    return {'samples':n,'bucket_counts':buckets,'bucket_percentages':[value/n for value in buckets],'mean':statistics.mean(values),'median':statistics.median(values),'standard_deviation':statistics.pstdev(values),'center_window_percentage':sum(.4<=value<.6 for value in values)/n,'exact_middle_percentage':exact}

def _v1_position_audit():
    result={}
    for split in SPLITS:
        groups=defaultdict(list)
        for row in read_jsonl(V1_PROCESSED/f'splits/{split}_localization.jsonl'): groups[row['sample_id']].append(row)
        converted={}
        for sample_id,rows in groups.items():
            start=min(r['line_number'] for r in rows); end=max(r['line_number'] for r in rows)
            converted[sample_id]=[{**r,'region_start_line':start,'region_end_line':end} for r in rows]
        result[split]=_position_summary(converted)
    return result

def prepare_v2():
    allowed=_allowed_ids(); targets=_source_records(allowed)
    found={}; ambiguous=set(); outcomes=Counter()
    expected={'condition','bug','fix','post_conditional','bugType','buggy_commit','repository_url'}
    for path in sorted((RAW/'sstub_input_no_comments').glob('*.parquet')):
        parquet=pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64,columns=sorted(expected)):
            for raw in batch.to_pylist():
                if any(not isinstance(raw[k],str) for k in expected): continue
                bug=normalize_text_field(raw['bug']); fixed=normalize_text_field(raw['fix'])
                key=alternate_key(raw['repository_url'],raw['buggy_commit'],bug,fixed,raw['bugType'].strip())
                target=targets.get(key)
                if not target: continue
                full=(raw['condition']+raw['bug']+raw['post_conditional']).replace('\r\n','\n').replace('\r','\n')
                prefix=raw['condition'].replace('\r\n','\n').replace('\r','\n'); bug_index=prefix.count('\n')
                if bug_index+1!=target['buggy_line_number'] or len(raw['condition'])!=target['bug_node_start']:
                    outcomes['verification_mismatch']+=1; continue
                source_hash=digest(full)
                if key in found and found[key]['source_hash']!=source_hash: ambiguous.add(key); found.pop(key,None); continue
                if key not in ambiguous: found[key]={**target,'source_hash':source_hash,'lines':full.split('\n'),'bug_index':bug_index}
    generated={split:[] for split in SPLITS}; source_regions={split:{} for split in SPLITS}
    for key,record in found.items():
        region=natural_region(record['lines'],record['bug_index'],record['sample_id'])
        rows=candidates(record,region,record['split'])
        if rows:
            generated[record['split']].append((record,rows)); source_regions[record['split']][record['sample_id']]=rows
        else: outcomes['no_valid_candidate_group']+=1
    # Purge complete later-split groups if an exact model input appeared earlier.
    seen=set(); exclusions=[]; final={}
    for split in SPLITS:
        rows=[]; kept={}
        for record,group in generated[split]:
            hashes={digest(model_text(row)) for row in group}
            if len(hashes)!=len(group):
                exclusions.append({'sample_id':record['sample_id'],'split':split,'reason':'duplicate_model_input_within_candidate_group'}); continue
            if hashes&seen:
                exclusions.append({'sample_id':record['sample_id'],'split':split,'reason':'exact_v2_model_input_overlaps_earlier_split'}); continue
            rows.extend(group); kept[record['sample_id']]=group; seen.update(hashes)
        final[split]=rows; source_regions[split]=kept
        write_jsonl(PROCESSED/f'splits/{split}_localization.jsonl',rows)
    positions={split:_position_summary(source_regions[split]) for split in SPLITS}
    save_json(REPORTS/'metrics/position_distribution.json',positions)
    random_results={split:random_baseline(final[split]) for split in SPLITS}
    position_results={split:position_baseline(final[split]) for split in SPLITS}
    save_json(REPORTS/'metrics/random_baseline.json',random_results); save_json(REPORTS/'metrics/position_baseline.json',position_results)
    repositories={split:{row['repository'] for row in final[split]} for split in SPLITS}
    input_hashes={split:{digest(model_text(row)) for row in final[split]} for split in SPLITS}
    integrity={'repository_overlap':all(not(repositories[a]&repositories[b]) for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]),'sample_id_overlap':all(not(set(source_regions[a])&set(source_regions[b])) for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]),'model_input_overlap':all(not(input_hashes[a]&input_hashes[b]) for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]),'duplicate_model_inputs_within_groups':False,'one_positive_per_group':all(sum(r['label'] for r in group)==1 for groups in source_regions.values() for group in groups.values()),'forbidden_predictive_fields_absent':all(not({'fixed_statement','patch','buggy_line_number','candidate_index','relative_position'}&set(row)) for rows in final.values() for row in rows),'excluded_groups':len(exclusions),'exclusion_reasons':dict(Counter(row['reason'] for row in exclusions))}
    failures=[]
    for split in ['validation','test']:
        if positions[split]['center_window_percentage']>MIDDLE_BUCKET_LIMIT: failures.append(f'{split} middle bucket exceeds {MIDDLE_BUCKET_LIMIT}')
        if max(positions[split]['bucket_percentages'])>SINGLE_BUCKET_LIMIT: failures.append(f'{split} positional bucket exceeds {SINGLE_BUCKET_LIMIT}')
    if position_results['test']['mrr']>POSITION_MRR_LIMIT: failures.append(f'test position-only MRR exceeds {POSITION_MRR_LIMIT}')
    if not all(integrity[k] for k in ['repository_overlap','sample_id_overlap','model_input_overlap','one_positive_per_group','forbidden_predictive_fields_absent']): failures.append('leakage or integrity check failed')
    counts={split:{'source_bugs':len(source_regions[split]),'candidate_lines':len(final[split]),'positives':sum(r['label'] for r in final[split]),'negatives':sum(not r['label'] for r in final[split]),'positive_negative_ratio':sum(r['label'] for r in final[split])/max(1,sum(not r['label'] for r in final[split])),'context_length_lines':{'min':min(r['region_end_line']-r['region_start_line']+1 for r in final[split]),'max':max(r['region_end_line']-r['region_start_line']+1 for r in final[split]),'mean':statistics.mean(r['region_end_line']-r['region_start_line']+1 for r in final[split])}} for split in SPLITS}
    gate={'passed':not failures,'training_allowed':False,'requires_user_review':True,'decision':'STOP_BEFORE_GPU_TRAINING','failures':failures,'thresholds':{'position_test_mrr_max':POSITION_MRR_LIMIT,'middle_bucket_max':MIDDLE_BUCKET_LIMIT,'single_bucket_max':SINGLE_BUCKET_LIMIT},'counts':counts,'integrity':integrity,'position_test':position_results['test'],'random_test':random_results['test']['mean']}
    save_json(REPORTS/'metrics/pretraining_gate.json',gate)
    metadata={'version':'2.0.0','source':'v1 split membership plus rejoined untouched full pre-fix source','v1_artifacts_modified':False,'counts':counts,'integrity':integrity,'processed_sha256':{str((PROCESSED/f'splits/{s}_localization.jsonl').relative_to(PROCESSED)):file_digest(PROCESSED/f'splits/{s}_localization.jsonl') for s in SPLITS},'ambiguous_keys':len(ambiguous),'outcomes':dict(outcomes),'context_policy':'Natural containing method, else containing block; regions over 61 lines receive deterministic hash-assigned debiased crops. Candidate order is seeded and label-independent.','training_negative_policy':'Six high-overlap executable hard negatives; evaluation uses all eligible lines.','model_feature_whitelist':['candidate_line','context']}
    save_json(PROCESSED/'metadata.json',metadata)
    v1=_v1_position_audit(); save_json(REPORTS/'metrics/v1_position_distribution.json',v1)
    _write_audit(v1,positions,position_results,random_results,gate,counts,integrity)
    return gate

def _write_audit(v1,v2,position,random_results,gate,counts,integrity):
    text='# CodeLocator v2 position-bias audit\n\n## Root cause in v1\n\nv1 first extracted a 31-line source window centered on the known bug (15 lines before and after). It then removed candidates near artificial crop edges. Consequently, 94.10% of validation and 94.70% of test bugs occupied the middle relative-position bucket. Candidate-specific seven-line model contexts were symmetric too, while the ranking candidate set retained the centered-source construction. The position-only diagnostic therefore ranked the known midpoint and achieved test MRR 0.8763.\n\n## Actual distributions\n\nBuckets are `[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]`.\n\n```json\n'+json.dumps({'v1':v1,'v2':v2},indent=2)+'\n```\n\n## v2 construction\n\nFull pre-fix sources were rejoined using the same exact repository, parent-commit, category, buggy-fragment and fixed-fragment identity checks. v1 split membership was preserved. v2 chooses the containing method when detected, otherwise a containing block. Regions longer than 61 lines use a deterministic hash-assigned crop position across five target buckets. Training keeps one view and six executable hard negatives chosen by identifier/operator similarity; validation and test retain all eligible lines. Candidate rows are deterministically shuffled. Model inputs contain only candidate code plus shared natural-region code.\n\n## Baselines and gate\n\n```json\n'+json.dumps({'counts':counts,'position':position,'random':random_results,'integrity':integrity,'gate':gate},indent=2)+'\n```\n\nGPU training was not started. The gate always stops for user inspection, and it also records any threshold failures.\n'
    REPORTS.mkdir(parents=True,exist_ok=True); (REPORTS/'POSITION_BIAS_AUDIT.md').write_text(text,encoding='utf-8')
