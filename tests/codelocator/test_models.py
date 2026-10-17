import numpy as np
import pytest
from codelocator.evaluate import classification_metrics,ranking_metrics
from codelocator.inference import CodeLocator
from codelocator.config import MODELS

def test_classification_metrics():
    result=classification_metrics([0,1],np.array([[.8,.2],[.1,.9]]),['A','B'])
    assert result['accuracy']==1 and result['macro_f1']==1

def test_ranking_metrics():
    rows=[{'sample_id':'a','line_number':1,'label':0},{'sample_id':'a','line_number':2,'label':1}]
    metrics=ranking_metrics(rows,[.9,.8]); assert metrics['mrr']==.5 and metrics['top_1']==0 and metrics['top_3']==1

@pytest.mark.parametrize('value',[None,'','  ',12,'x\x00y'])
def test_invalid_input(value):
    model=object.__new__(CodeLocator)
    with pytest.raises(ValueError): model.analyze_code(value)

def test_baseline_loading_and_inference():
    model=CodeLocator('baseline')
    assert model.baseline.named_steps['tfidf'].lowercase is False
    result=model.analyze_code('total = price + discount;')
    assert result['bug_probability'] is None and result['prediction']=='Defect presence not assessed'
    assert 0<=result['bug_type_confidence']<=1
    assert result['predicted_bug_type'] in model.baseline.named_steps['clf'].classes_
    assert isinstance(result['suspicious_lines'],list)

def test_tokenizer_preprocessing():
    path=MODELS/'pretrained/codebert-base'
    if not (path/'source_manifest.json').exists(): pytest.skip('Pretrained tokenizer download has not completed')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    plus=tokenizer('a + b',max_length=256,truncation=True)['input_ids']
    minus=tokenizer('a - b',max_length=256,truncation=True)['input_ids']
    assert plus!=minus and len(plus)<=256

@pytest.mark.parametrize('name',['bug_type_classifier','line_localizer'])
def test_transformer_loading(name):
    path=MODELS/name
    if not (path/'training_metadata.json').exists(): pytest.skip(f'{name} training has not completed')
    model,tokenizer,device,cfg=CodeLocator.load_transformer(path)
    assert cfg['max_length']<=256 and model.config.num_labels>=2
