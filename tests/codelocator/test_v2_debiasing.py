import json
import hashlib
from pathlib import Path
from codelocator_v2.baselines import position_baseline,random_baseline
from codelocator_v2.context import candidates,model_text,natural_region

def sample(sample_id='sample'):
    lines=['class A {','  void f() {']+[f'    value{i} = call(value{i});' for i in range(75)]+['  }','}']
    return lines,{'sample_id':sample_id,'repository':'owner/repo','buggy_line_number':32}

def test_debiased_crop_is_reproducible_and_varies_by_sample():
    lines,_=sample()
    first=natural_region(lines,31,'stable')
    assert first==natural_region(lines,31,'stable')
    starts={natural_region(lines,31,f'sample-{i}')['start'] for i in range(30)}
    # Source/method boundaries can collapse requested buckets; variation must
    # still exist wherever multiple valid crop starts are available.
    assert len(starts)>=3

def test_candidate_order_does_not_fix_positive_index_or_leak_features():
    indices=set()
    for i in range(20):
        lines,record=sample(f'sample-{i}')
        region=natural_region(lines,31,record['sample_id'])
        rows=candidates(record,region,'validation')
        indices.add(next(j for j,row in enumerate(rows) if row['label']))
        assert all('BUG' not in model_text(row) for row in rows)
        assert all(not ({'fixed_statement','patch','buggy_line_number','candidate_index','relative_position'}&set(row)) for row in rows)
    assert len(indices)>1

def test_random_baseline_is_reproducible_and_position_baseline_is_correct():
    rows=[{'sample_id':'a','line_number':n,'region_start_line':1,'region_end_line':5,'label':int(n==3)} for n in range(1,6)]
    assert random_baseline(rows)==random_baseline(rows)
    metrics=position_baseline(rows)
    assert metrics['top_1']==1 and metrics['mrr']==1

def test_v2_gate_artifact_blocks_automatic_training_when_present():
    path=Path('reports/codelocator_v2/metrics/pretraining_gate.json')
    if path.exists():
        gate=json.loads(path.read_text())
        assert gate['training_allowed'] is False
        assert gate['decision']=='STOP_BEFORE_GPU_TRAINING'

def test_v1_artifacts_match_preservation_manifest():
    manifest=Path('reports/codelocator_v2/metrics/v1_artifact_manifest.json')
    if manifest.exists():
        for name,expected in json.loads(manifest.read_text()).items():
            path=Path(name)
            assert path.stat().st_size==expected['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']

def test_v2_inference_artifacts_are_complete():
    model=Path('models/codelocator_v2/line_localizer')
    if model.exists():
        assert all((model/name).is_file() for name in ['model.safetensors','training_metadata.json','evaluation.json','tokenizer.json'])
