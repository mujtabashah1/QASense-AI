"""Memory-conscious CodeBERT fine-tuning; no fixed code in input."""
import json
import math
import random
from pathlib import Path
import numpy as np
from .config import TRAINING, MODELS, REPORTS, SEED, PROCESSED
from .utils import save_json
from .evaluate import classification_metrics

def hardware_probe():
    import torch
    info={'pytorch':torch.__version__,'cuda_available':torch.cuda.is_available(),'compiled_cuda':torch.version.cuda}
    if info['cuda_available']:
        info.update(gpu=torch.cuda.get_device_name(0),vram_gb=torch.cuda.get_device_properties(0).total_memory/1024**3,capability=list(torch.cuda.get_device_capability(0)))
        # Prove a CUDA kernel works, rather than relying only on enumeration.
        value=torch.ones(2,device='cuda').sum().item()
        info['cuda_kernel_check']=value==2
    save_json(REPORTS/'metrics/hardware.json',info)
    print(json.dumps(info,indent=2),flush=True)
    return info

def format_bug_type(code): return code

def configure_trainable_layers(model, final_layers=4):
    encoder=model.roberta.encoder.layer
    if not 0<final_layers<=len(encoder): raise ValueError('Invalid number of trainable encoder layers')
    for parameter in model.parameters(): parameter.requires_grad=False
    first=len(encoder)-final_layers
    for layer in encoder[first:]:
        for parameter in layer.parameters(): parameter.requires_grad=True
    for parameter in model.classifier.parameters(): parameter.requires_grad=True
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'encoder_layer_indexing':'zero-based','frozen_embeddings':True,'frozen_encoder_layers':list(range(first)),'trained_encoder_layers':list(range(first,len(encoder))),'trained_head':'classifier (dense and out_proj)','trainable_parameters':trainable,'total_parameters':sum(p.numel() for p in model.parameters()),'policy':'Only final four encoder layers and task head; no training dataset or split reduction'}

def fit_transformer(texts,targets,validation_texts,validation_targets,names,output_dir,task,config=None,selection_function=None):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from .batching import training_groups
    cfg=dict(TRAINING); cfg.update(config or {})
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu': torch.set_num_threads(4)
    prepared=MODELS/'pretrained/codebert-base'
    base_reference=str(prepared) if (prepared/'source_manifest.json').is_file() else cfg['pretrained_model']
    tokenizer=AutoTokenizer.from_pretrained(base_reference,cache_dir=MODELS/'pretrained',use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(base_reference,cache_dir=MODELS/'pretrained',num_labels=len(names),id2label={i:v for i,v in enumerate(names)},label2id={v:i for i,v in enumerate(names)},ignore_mismatched_sizes=True)
    layer_policy=configure_trainable_layers(model,cfg['train_final_encoder_layers'])
    print('Layer policy: '+json.dumps(layer_policy),flush=True)
    save_json(REPORTS/f'metrics/{task}_layer_policy.json',layer_policy)
    model.to(device)
    # Non-reentrant checkpointing supports frozen inputs; no embedding gradient hook
    # is needed, so backward computation stops at the frozen lower encoder.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    encoded=tokenizer(texts,padding=False,truncation=True,max_length=cfg['max_length'])
    lengths=[len(v) for v in encoded['input_ids']]
    effective_batch=cfg['batch_size']*cfg['gradient_accumulation']
    token_budget=cfg['batch_size']*cfg['max_length']
    cfg.update(batching='length-aware, bounded padded tokens; pretokenized once',effective_batch_size=effective_batch,token_budget=token_budget)
    counts=np.bincount(targets,minlength=len(names)); weights=np.sqrt(max(counts.sum(),1)/np.maximum(counts,1)); weights/=weights.mean()
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),reduction='sum')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    steps_per_epoch=math.ceil(len(texts)/effective_batch)
    total_steps=steps_per_epoch*cfg['epochs']; warmup=max(1,int(total_steps*cfg['warmup_ratio']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: min((step+1)/warmup,max(0,(total_steps-step)/max(1,total_steps-warmup))))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=-1; stale=0; history=[]
    for epoch in range(cfg['epochs']):
        model.train(); optimizer.zero_grad(); total_loss=0
        for i,(group,microbatches) in enumerate(training_groups(lengths,SEED+epoch,effective_batch,token_budget)):
            denominator=float(sum(weights[targets[index]] for index in group)); group_loss=0
            for indices in microbatches:
                batch=tokenizer.pad([{k:values[index] for k,values in encoded.items()} for index in indices],padding=True,return_tensors='pt')
                batch={k:v.to(device) for k,v in batch.items()}; y=torch.tensor([targets[index] for index in indices],dtype=torch.long,device=device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    logits=model(**batch).logits
                    loss=loss_fn(logits.float(),y)/denominator
                group_loss+=loss.item(); scaler.scale(loss).backward()
            total_loss+=group_loss
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
            if (i+1)%100==0: print(f'{task} epoch {epoch+1}: {i+1}/{steps_per_epoch} optimizer steps',flush=True)
        probabilities=predict_transformer(model,tokenizer,validation_texts,cfg,device)
        metrics=classification_metrics(validation_targets,probabilities,names)
        selection_score=selection_function(probabilities) if selection_function else metrics['macro_f1']
        row={'epoch':epoch+1,'training_loss':total_loss/steps_per_epoch,'validation':metrics}; history.append(row)
        print(f'{task} epoch {epoch+1}: validation macro F1 {metrics["macro_f1"]:.4f}',flush=True)
        if selection_score>best:
            best=selection_score; stale=0; model.save_pretrained(output_dir,safe_serialization=True); tokenizer.save_pretrained(output_dir)
            save_json(Path(output_dir)/'training_metadata.json',{'task':task,'config':cfg,'layer_policy':layer_policy,'device':str(device),'train_samples':len(texts),'validation_samples':len(validation_texts),'label_mapping':{v:i for i,v in enumerate(names)},'best_validation_score':best,'selection_metric':'MRR' if selection_function else 'macro_f1','class_weights':weights.tolist(),'input_policy':'No fixed code, patch, repository, commit, target line number, labels or change markers included as features','pretrained_revision':json.loads((prepared/'source_manifest.json').read_text())['revision'] if (prepared/'source_manifest.json').exists() else getattr(model.config,'_commit_hash',None)})
        else: stale+=1
        save_json(REPORTS/f'metrics/{task}_history.json',history)
        if stale>=cfg['early_stopping_patience']: break
    del model,optimizer,scaler
    if device.type=='cuda': torch.cuda.empty_cache()
    model=AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    return model,tokenizer,cfg,device

def predict_transformer(model,tokenizer,texts,cfg,device):
    import torch
    from .batching import token_batches
    if not texts: raise ValueError('No transformer inputs')
    model.eval(); tokenized=tokenizer(texts,padding=False,truncation=True,max_length=cfg['max_length'])
    lengths=[len(v) for v in tokenized['input_ids']]
    order=sorted(range(len(texts)),key=lambda i:lengths[i])
    output=np.empty((len(texts),model.config.num_labels),dtype=np.float32)
    with torch.inference_mode():
        for indices in token_batches(order,lengths,cfg['batch_size']*cfg['max_length']):
            encoded=tokenizer.pad([{k:values[index] for k,values in tokenized.items()} for index in indices],padding=True,return_tensors='pt')
            encoded={k:v.to(device) for k,v in encoded.items()}
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'): logits=model(**encoded).logits
            output[indices]=torch.softmax(logits.float(),dim=-1).cpu().numpy()
    return output

def fit_with_oom_recovery(*args,**kwargs):
    import torch
    attempts=[{}, {'batch_size':1,'gradient_accumulation':16}, {'batch_size':1,'gradient_accumulation':16,'max_length':192}, {'batch_size':1,'gradient_accumulation':16,'max_length':128}]
    for changes in attempts:
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        try: return fit_transformer(*args,**kwargs,config=changes)
        except torch.cuda.OutOfMemoryError:
            import gc
            gc.collect(); torch.cuda.empty_cache()
            print('CUDA OOM: retrying with smaller batch/sequence; gradient checkpointing remains enabled.',flush=True)
    raise RuntimeError('CUDA OOM after batch 1, accumulation 16, lengths 256/192/128 and gradient checkpointing')

def train_bug_type(splits,mapping):
    names=sorted(mapping['label2id'],key=mapping['label2id'].get)
    texts=lambda split:[format_bug_type(r['buggy_statement']) for r in splits[split]]
    ys=lambda split:[mapping['label2id'][r['bug_type']] for r in splits[split]]
    model,tokenizer,cfg,device=fit_with_oom_recovery(texts('train'),ys('train'),texts('validation'),ys('validation'),names,MODELS/'bug_type_classifier','bug_type')
    results={n:classification_metrics(ys(n),predict_transformer(model,tokenizer,texts(n),cfg,device),names) for n in ['validation','test']}
    save_json(REPORTS/'metrics/codebert_bug_type.json',results)
    save_json(MODELS/'bug_type_classifier/evaluation.json',results)
    save_json(MODELS/'bug_type_classifier/label_mapping.json',mapping)
    del model
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results
