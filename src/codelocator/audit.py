import json
import pandas as pd
from .config import REPORTS, PROCESSED
from .data_loader import load_primary

def audit_dataframe(df):
    text_fields=['sourceBeforeFix','sourceAfterFix','fixPatch']
    audit={'rows':len(df),'columns':len(df.columns),'column_names':list(df.columns),'dtypes':df.dtypes.astype(str).to_dict(),'missing_values':{c:int(df[c].isna().sum()) for c in df},'null_percentages':{c:float(df[c].isna().mean()*100) for c in df},'empty_strings':{c:int(df[c].map(lambda v:isinstance(v,str) and not v.strip()).sum()) for c in df},'exact_duplicate_rows':int(df.duplicated().sum()),'duplicate_buggy_statements':int(df.sourceBeforeFix.duplicated().sum()),'duplicate_fixed_statements':int(df.sourceAfterFix.duplicated().sum()),'duplicate_pairs':int(df.duplicated(['sourceBeforeFix','sourceAfterFix']).sum()),'duplicate_patches':int(df.fixPatch.duplicated().sum()),'unique_projects':int(df.projectName.nunique()),'repository_field':'projectName is the owner.repository identifier; no separate repository URL in primary schema','unique_repositories':int(df.projectName.nunique()),'unique_commits':int(df.fixCommitSHA1.nunique()),'unique_files':int(df[['projectName','bugFilePath']].drop_duplicates().shape[0]),'unique_bug_types':int(df.bugType.nunique()),'identical_buggy_fixed':int(df.sourceBeforeFix.eq(df.sourceAfterFix).sum()),'invalid_line_numbers':int((pd.to_numeric(df.bugLineNum,errors='coerce').fillna(0)<=0).sum()),'line_number_distribution':{k:float(v) for k,v in df.bugLineNum.describe().items()},'source_length_distributions':{c:{k:float(v) for k,v in df[c].fillna('').str.len().describe().items()} for c in text_fields},'unusable_source_records':int((df.sourceBeforeFix.isna() | df.sourceBeforeFix.fillna('').str.strip().eq('')).sum()),'source_interpretation':'sourceBeforeFix/sourceAfterFix are AST code fragments, not complete files. fixPatch is a unified diff, excluded from predictive inputs.'}
    distribution=df.bugType.value_counts(dropna=False).rename_axis('bug_type').reset_index(name='count')
    distribution['percentage']=distribution['count']/len(df)*100
    audit['bug_type_distribution']=distribution.to_dict('records')
    audit['rare_classes']={str(n):distribution.loc[distribution['count']<n,'bug_type'].tolist() for n in [100,50,20]}
    audit['malformed_records']=int(df[list(['projectName','bugType','fixCommitSHA1','bugFilePath','sourceBeforeFix','sourceAfterFix'])].isna().any(axis=1).sum())
    from .utils import save_json
    save_json(REPORTS/'metrics/data_audit.json',audit)
    save_json(PROCESSED/'audit/data_audit.json',audit)
    distribution.to_csv(REPORTS/'tables/bug_type_distribution.csv',index=False)
    (REPORTS/'data_audit.md').write_text('# CodeLocator data audit\n\nSchema discovered from the real primary parquet. No raw files were modified.\n\n```json\n'+json.dumps(audit,indent=2)+'\n```\n',encoding='utf-8')
    return audit

def main(): return audit_dataframe(load_primary())
