"""Short throughput measurement only; never saves benchmark weights or metrics."""
import gc
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator.config import MODELS,PROCESSED,REPORTS,TRAINING
from codelocator.preprocessing import validate_processed
from codelocator.train_bug_type import configure_trainable_layers
from codelocator.batching import training_groups
from codelocator.context_builder import format_candidate
from codelocator.utils import read_jsonl,save_json

def benchmark(texts,labels,num_labels,steps=30,token_budget=512):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    tokenizer=AutoTokenizer.from_pretrained(MODELS/'pretrained/codebert-base',local_files_only=True)
    encoded=tokenizer(texts,truncation=True,max_length=256,padding=False)
    lengths=[len(x) for x in encoded['input_ids']]
    model=AutoModelForSequenceClassification.from_pretrained(MODELS/'pretrained/codebert-base',local_files_only=True,num_labels=num_labels,ignore_mismatched_sizes=True).cuda()
    policy=configure_trainable_layers(model,4)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda'); times=[]; torch.cuda.reset_peak_memory_stats()
    for i,(group,batches) in enumerate(training_groups(lengths,42,16,token_budget)):
        if i>=steps+3: break
        torch.cuda.synchronize(); start=time.perf_counter(); optimizer.zero_grad()
        for indices in batches:
            batch=tokenizer.pad([{k:values[j] for k,values in encoded.items()} for j in indices],padding=True,return_tensors='pt'); batch={k:v.cuda() for k,v in batch.items()}
            targets=torch.tensor([labels[j] for j in indices],device='cuda')
            with torch.autocast('cuda',dtype=torch.float16): loss=torch.nn.functional.cross_entropy(model(**batch).logits,targets,reduction='sum')/len(group)
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        if i>=3: times.append(time.perf_counter()-start)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    result={'steps_measured':len(times),'token_budget':token_budget,'gradient_policy_verified':True,'mean_seconds_per_optimizer_step':sum(times)/len(times),'peak_gpu_allocated_gb':torch.cuda.max_memory_allocated()/1024**3,'layer_policy':policy,'note':'Temporary runtime benchmark; no weights saved, no evaluation results, no change to prepared data or splits.'}
    print(result,flush=True)
    del model,optimizer,scaler; gc.collect(); torch.cuda.empty_cache()
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--token-budget',type=int,default=512); parser.add_argument('--steps',type=int,default=30); args=parser.parse_args()
    splits,metadata,mapping=validate_processed()
    rows=splits['train']
    classifier=benchmark([r['buggy_statement'] for r in rows],[mapping['label2id'][r['bug_type']] for r in rows],len(mapping['label2id']),args.steps,args.token_budget)
    candidates=read_jsonl(PROCESSED/'splits/train_localization.jsonl')
    localizer=benchmark([format_candidate(r['candidate_line'],r['context']) for r in candidates],[r['label'] for r in candidates],2,args.steps,args.token_budget)
    save_json(REPORTS/f'metrics/layer_runtime_benchmark_{args.token_budget}.json',{'final_four_bug_type':classifier,'final_four_localizer':localizer,'full_encoder_bug_type_observation':{'steps_observed':500,'approximate_elapsed_seconds':500},'planned_epochs':3})
