import pandas as pd
import pytest
from codelocator.data_loader import load_parquet,load_primary
from codelocator.cleaning import normalize_text_field,normalize_bug_type,is_valid_bug_pair,make_sample_id,clean_dataframe
from codelocator.context_builder import extract_context,candidate_lines,format_candidate
from codelocator.splitting import verify_splits
from codelocator.preprocessing import validate_processed
from codelocator.config import RAW,PROCESSED
from codelocator.utils import file_digest,read_jsonl,digest

def test_real_loader():
    frame=load_primary()
    assert frame.shape==(63923,14) and 'sourceBeforeFix' in frame

def test_loader_missing_and_empty(tmp_path,monkeypatch):
    with pytest.raises(FileNotFoundError): load_parquet(tmp_path/'missing')
    file=tmp_path/'empty'; file.touch()
    monkeypatch.setattr(pd,'read_parquet',lambda p:pd.DataFrame())
    with pytest.raises(ValueError): load_parquet(file)

@pytest.mark.parametrize('value',[None,'','  ',float('nan'),2])
def test_missing_text(value): assert normalize_text_field(value) is None

def test_conservative_cleaning():
    source='  if (Foo != null) {\r\n    a += 1;\r\n}'
    assert normalize_text_field(source)=='  if (Foo != null) {\n    a += 1;\n}'
    assert normalize_bug_type(' CHANGE_OPERATOR ')=='CHANGE_OPERATOR'
    assert is_valid_bug_pair('a + b','a - b')
    assert not is_valid_bug_pair('a + b','a + b')

def test_sample_ids_and_duplicates(monkeypatch,tmp_path):
    import codelocator.cleaning as module
    monkeypatch.setattr(module,'REPORTS',tmp_path)
    (tmp_path/'tables').mkdir()
    frame=load_primary().head(1)
    records,summary=clean_dataframe(pd.concat([frame,frame],ignore_index=True))
    assert len(records)==1 and summary['dropped']==1
    assert make_sample_id(records[0])==records[0]['sample_id']
    changed=dict(records[0],bug_type='OTHER'); assert make_sample_id(changed)!=records[0]['sample_id']

def test_context_line_preservation():
    context=extract_context(['a','b','c','d'],2,before=1,after=1)
    assert context['context']=='b\nc\nd' and context['context_start_line']==2 and context['bug_index_in_context']==1
    with pytest.raises(ValueError): extract_context(['a'],1)

def test_no_artificial_crop_edges():
    record={'context_verified':True,'context_lines':['x'+str(i) for i in range(31)],'context_start_line':100,'source_line_count':200,'sample_id':'s','repository':'r','buggy_line_number':115}
    rows=candidate_lines(record)
    assert len(rows)==25 and min(r['line_number'] for r in rows)==103 and max(r['line_number'] for r in rows)==127
    assert sum(r['label'] for r in rows)==1
    assert all(len(r['context'].split('\n'))==7 for r in rows)
    assert format_candidate('a + b','real\nsource')=='a + b\n\nreal\nsource'

def test_processed_quality_and_leakage():
    splits,metadata,mapping=validate_processed()
    assert verify_splits(splits)
    assert len(mapping['label2id'])==14
    assert metadata['context']['verified_samples']==12505
    for path,sha in metadata['raw_sha256'].items(): assert file_digest(RAW/path)==sha
    invalid={k:list(v) for k,v in splits.items()}; invalid['test']=[splits['train'][0]]
    with pytest.raises(ValueError): verify_splits(invalid)

def test_localization_input_leakage():
    seen=set()
    for name in ['train','validation','test']:
        rows=read_jsonl(PROCESSED/f'splits/{name}_localization.jsonl')
        hashes={digest(format_candidate(r['candidate_line'],r['context'])) for r in rows}
        assert not seen&hashes
        seen.update(hashes)
        from collections import Counter
        positives=Counter(r['sample_id'] for r in rows if r['label'])
        assert all(v==1 for v in positives.values())
        assert set(positives)=={r['sample_id'] for r in rows}
