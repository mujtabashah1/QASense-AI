"""Verify the real pretrained tokenizer/model without claiming training results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator.config import MODELS,REPORTS
from codelocator.preprocessing import validate_processed
from codelocator.utils import save_json

if __name__=='__main__':
    _,_,mapping=validate_processed()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    path=MODELS/'pretrained/codebert-base'
    names=sorted(mapping['label2id'],key=mapping['label2id'].get)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,num_labels=len(names),id2label={i:v for i,v in enumerate(names)},label2id={v:i for i,v in enumerate(names)},ignore_mismatched_sizes=True)
    torch.set_num_threads(4); model.eval()
    batch=tokenizer(['total = price + discount;','if (item != null) return item;'],padding=True,truncation=True,max_length=256,return_tensors='pt')
    with torch.inference_mode(): outputs=model(**batch).logits
    assert tuple(outputs.shape)==(2,len(names)) and torch.isfinite(outputs).all()
    result={'pretrained_weights_loaded':True,'tokenizer_loaded':True,'cpu_forward_shape':list(outputs.shape),'classification_head':'new random initialization; this is a compatibility check, not a trained classifier or evaluation'}
    save_json(REPORTS/'metrics/preflight.json',result)
    print(result)
