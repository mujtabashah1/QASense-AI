"""Verified source reconstruction from the included alternate configuration."""
from collections import Counter
from .utils import digest, repository_key
from .config import RAW, TRAINING
from .cleaning import normalize_text_field

def extract_context(lines,bug_index,before=15,after=15):
    if not isinstance(bug_index,int) or not 0<=bug_index<len(lines): raise ValueError('Invalid zero-based source index')
    start=max(0,bug_index-before); end=min(len(lines),bug_index+after+1)
    return {'context':'\n'.join(lines[start:end]),'context_start_line':start+1,'bug_index_in_context':bug_index-start,'context_lines':lines[start:end]}

def alternate_key(repo,commit,bug,fixed,label):
    return digest('\0'.join([repository_key(repo),commit,bug,fixed,label]))

def build_context_index():
    import pyarrow.parquet as pq
    index={}; schema_info=[]; count=0
    for path in sorted((RAW/'sstub_input_no_comments').glob('*.parquet')):
        file=pq.ParquetFile(path)
        columns=file.schema_arrow.names
        expected={'condition','bug','fix','post_conditional','bugType','buggy_commit','repository_url'}
        if not expected <= set(columns): raise ValueError('Alternate context schema changed')
        schema_info.append({'file':path.name,'columns':columns,'rows':file.metadata.num_rows})
        for batch in file.iter_batches(batch_size=64,columns=sorted(expected)):
            for r in batch.to_pylist():
                count+=1
                if any(not isinstance(r[k],str) for k in expected): continue
                bug=normalize_text_field(r['bug']); fixed=normalize_text_field(r['fix'])
                if not bug or not fixed: continue
                prefix=r['condition']; suffix=r['post_conditional']
                full=(prefix+ r['bug']+suffix).replace('\r\n','\n').replace('\r','\n')
                prefix_normal=prefix.replace('\r\n','\n').replace('\r','\n')
                bug_index=prefix_normal.count('\n')
                lines=full.split('\n')
                context=extract_context(lines,bug_index,TRAINING['source_radius'],TRAINING['source_radius'])
                context.update(prefix_char_count=len(prefix),original_line=bug_index+1,source_hash=digest(full),source_line_count=len(lines))
                key=alternate_key(r['repository_url'],r['buggy_commit'],bug,fixed,r['bugType'].strip())
                if key not in index: index[key]=context
                elif index[key] is not None and index[key]['source_hash']!=context['source_hash']: index[key]=None
    return index,{'alternate_rows':count,'schemas':schema_info,'index_entries':len(index),'ambiguous_source_keys':sum(v is None for v in index.values())}

def attach_context(records):
    index,info=build_context_index(); outcomes=Counter()
    for r in records:
        r.update(context=None,context_lines=None,context_start_line=None,context_verified=False)
        key=alternate_key(r['repository'],r['parent_commit'],r['buggy_statement'],r['fixed_statement'],r['bug_type'])
        candidate=index.get(key)
        reason='no_unique_context_join'
        if candidate:
            if candidate['original_line']!=r['buggy_line_number']: reason='line_number_mismatch'
            elif candidate['prefix_char_count']!=r['bug_node_start']: reason='character_offset_mismatch'
            elif '\n' in r['buggy_statement']: reason='multiline_fragment_not_single_line_label'
            elif len(candidate['context_lines'])<3: reason='insufficient_genuine_context'
            elif r['buggy_statement'].strip() not in candidate['context_lines'][candidate['bug_index_in_context']]: reason='bug_fragment_not_at_labelled_line'
            else:
                r.update(candidate); r['context_verified']=True; reason='verified_full_source_join'
        outcomes[reason]+=1
    info.update(outcomes=dict(outcomes),verified_samples=outcomes['verified_full_source_join'],genuine_context_available=outcomes['verified_full_source_join']>0,localization_scientifically_supported=outcomes['verified_full_source_join']>=100,decision='Only complete pre-fix sources reconstructed from condition + bug + post_conditional, joined by repository, parent commit, category and exact buggy/fixed fragments, with matching line and character offset. Patches are never predictive inputs. No fabricated context.')
    return info

def candidate_lines(record,negatives=None):
    if not record['context_verified']: return []
    result=[]; radius=TRAINING['candidate_radius']; lines=record['context_lines']; first=record['context_start_line']
    for i,line in enumerate(lines):
        if not line.strip(): continue
        # Do not expose artificial crop edges as a shortcut for the bug position.
        if i<radius and first>1: continue
        if i>=len(lines)-radius and first+len(lines)-1<record['source_line_count']: continue
        window=extract_context(lines,i,radius,radius)
        result.append({'sample_id':record['sample_id'],'repository':record['repository'],'line_number':first+i,'candidate_line':line,'context':window['context'],'label':int(first+i==record['buggy_line_number'])})
    if negatives is not None:
        positive=[r for r in result if r['label']]; negative=[r for r in result if not r['label']]
        negative.sort(key=lambda r:digest(record['sample_id']+str(r['line_number'])))
        result=positive+negative[:negatives]
    return result

def format_candidate(candidate,context):
    # Identical format for positive/negative/inference; no line numbers or bug markers.
    return candidate+'\n\n'+context
