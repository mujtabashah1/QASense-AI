import json,math,os,random,time
from pathlib import Path
import numpy as np
from codelocator.batching import token_batches
from codelocator.evaluate import classification_metrics,ranking_metrics
from codelocator.train_bug_type import configure_trainable_layers
from codelocator.utils import file_digest,save_json
from .config import MODELS,PROCESSED,REPORTS,SEED
from .context import model_text

MAX_LENGTH=256
EFFECTIVE_BATCH=16

def micro_groups(lengths,epoch,physical_batch):
    rng=random.Random(SEED+epoch); indices=list(range(len(lengths))); rng.shuffle(indices); groups=[]
    for start in range(0,len(indices),256):
        bucket=sorted(indices[start:start+256],key=lambda i:lengths[i])
        groups.extend(bucket[i:i+EFFECTIVE_BATCH] for i in range(0,len(bucket),EFFECTIVE_BATCH))
    rng.shuffle(groups)
    for group in groups:
        yield group,list(token_batches(group,lengths,physical_batch*MAX_LENGTH,max_examples=physical_batch))

def load_rows():
    from codelocator.utils import read_jsonl
    return {name:read_jsonl(PROCESSED/f'splits/{name}_localization.jsonl') for name in ['train','validation','test']}

def split_identity():
    metadata=json.loads((PROCESSED/'metadata.json').read_text(encoding='utf-8'))
    actual={name:file_digest(PROCESSED/name) for name in metadata['processed_sha256']}
    if actual!=metadata['processed_sha256']: raise ValueError('V2 split checksum mismatch')
    return actual

def initialize_model(checkpointing):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    source=Path('models/codelocator/line_localizer')
    model=AutoModelForSequenceClassification.from_pretrained(source,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    policy=configure_trainable_layers(model,4)
    if checkpointing: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    return model,tokenizer,policy

def pad(tokenizer,encoded,indices,device):
    batch=tokenizer.pad([{key:values[index] for key,values in encoded.items()} for index in indices],padding=True,return_tensors='pt')
    return {key:value.to(device) for key,value in batch.items()}

def predict(model,tokenizer,encoded,lengths,physical_batch,device):
    import torch
    order=sorted(range(len(lengths)),key=lambda i:lengths[i]); output=np.empty((len(lengths),2),dtype=np.float32); model.eval()
    with torch.inference_mode():
        for indices in token_batches(order,lengths,physical_batch*MAX_LENGTH,max_examples=physical_batch):
            batch=pad(tokenizer,encoded,indices,device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'): logits=model(**batch).logits
            output[indices]=torch.softmax(logits.float(),dim=-1).cpu().numpy()
    return output

def evaluation(rows,probabilities):
    classification=classification_metrics([row['label'] for row in rows],probabilities,['CONTEXT_LINE','ANNOTATED_BUG_LINE'])
    return {'ranking':ranking_metrics(rows,probabilities[:,1]),'candidate_classification':classification,'bug_line':classification['per_class']['ANNOTATED_BUG_LINE']}

def atomic_torch_save(path,state):
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary); os.replace(temporary,path)

def rng_state():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])

def log(message):
    REPORTS.mkdir(parents=True,exist_ok=True)
    with (REPORTS/'training.log').open('a',encoding='utf-8') as handle: handle.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+message+'\n')
    print(message,flush=True)
