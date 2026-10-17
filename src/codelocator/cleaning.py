import json
import math
import numbers
from collections import Counter
import pandas as pd
from .utils import digest, repository_key
from .config import REPORTS

def normalize_text_field(value):
    if not isinstance(value,str) or not value.strip(): return None
    return value.replace('\r\n','\n').replace('\r','\n')

def normalize_bug_type(value):
    return value.strip() if isinstance(value,str) and value.strip() else None

def is_valid_bug_pair(buggy,fixed):
    return bool(buggy and fixed and buggy.strip() and fixed.strip() and buggy!=fixed)

def make_sample_id(record):
    keys=['repository','commit','file_path','buggy_line_number','bug_type','buggy_statement','fixed_statement']
    return digest(json.dumps([record[k] for k in keys],ensure_ascii=False,separators=(',',':')))

def clean_dataframe(df):
    records,drops,seen=[],[],set()
    for index,r in enumerate(df.to_dict('records')):
        buggy=normalize_text_field(r['sourceBeforeFix']); fixed=normalize_text_field(r['sourceAfterFix']); label=normalize_bug_type(r['bugType'])
        line=r['bugLineNum']
        reason=None
        if buggy is None: reason='missing_or_empty_buggy_code'
        elif fixed is None: reason='missing_or_empty_fixed_code'
        elif buggy==fixed: reason='identical_buggy_fixed_code'
        elif label is None: reason='missing_bug_type'
        elif not isinstance(line,numbers.Real) or not math.isfinite(line) or line<=0 or int(line)!=line: reason='invalid_line_number'
        elif any(not normalize_text_field(r[k]) for k in ['projectName','fixCommitSHA1','fixCommitParentSHA1','bugFilePath']): reason='missing_leakage_boundary'
        if reason:
            drops.append({'raw_row_index':index,'reason':reason}); continue
        record={'project':r['projectName'],'repository':repository_key(r['projectName']),'commit':r['fixCommitSHA1'],'parent_commit':r['fixCommitParentSHA1'],'file_path':r['bugFilePath'],'bug_type':label,'buggy_statement':buggy,'fixed_statement':fixed,'buggy_code':buggy,'fixed_code':fixed,'buggy_line_number':int(line),'bug_node_start':int(r['bugNodeStartChar']),'patch':normalize_text_field(r['fixPatch']),'raw_row_index':index}
        record['sample_id']=make_sample_id(record)
        record.update(buggy_hash=digest(buggy),fixed_hash=digest(fixed),pair_hash=digest(json.dumps([buggy,fixed])),patch_hash=digest(record['patch']) if record['patch'] else None)
        if record['sample_id'] in seen: drops.append({'raw_row_index':index,'reason':'exact_standardized_duplicate'}); continue
        seen.add(record['sample_id']); records.append(record)
    pd.DataFrame(drops,columns=['raw_row_index','reason']).to_csv(REPORTS/'tables/dropped_records.csv',index=False)
    counts=Counter(r['reason'] for r in drops)
    pd.DataFrame([{'reason':k,'count':v} for k,v in counts.items()]).to_csv(REPORTS/'tables/dropped_records_summary.csv',index=False)
    summary={'original_rows':len(df),'retained':len(records),'dropped':len(drops),'reasons':dict(counts)}
    (REPORTS/'cleaning_summary.md').write_text('# Conservative cleaning\n\nOnly line endings are normalized; identifiers, literals, case, operators, punctuation, indentation and line boundaries are preserved. Duplicate means identical standardized sample identity including the bug category. Different labels for the same fragment are retained and documented.\n\n'+json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return records,summary
