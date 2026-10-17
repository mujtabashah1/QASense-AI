import gc,json,subprocess,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from codelocator.utils import save_json
from codelocator_v2.config import REPORTS
from codelocator_v2.gate import require_reviewed_gate
from codelocator_v2.runtime import EFFECTIVE_BATCH,MAX_LENGTH,initialize_model,load_rows,micro_groups,pad
from codelocator_v2.context import model_text

def gpu_util_sampler(stop,values):
    command=[r'C:\Windows\System32\nvidia-smi.exe','--query-gpu=utilization.gpu','--format=csv,noheader,nounits']
    while not stop.wait(.5):
        try: values.append(float(subprocess.check_output(command,text=True,creationflags=0x08000000).strip()))
        except Exception: pass

def run(encoded,lengths,labels,physical,checkpointing,steps=15):
    import torch
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model,tokenizer,policy=initialize_model(checkpointing); model.cuda().train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda'); measured=[]; micro_times=[]; examples=0; micro_count=0; utils=[]; stop=threading.Event(); monitor=threading.Thread(target=gpu_util_sampler,args=(stop,utils),daemon=True); monitor.start()
    stable=True; error=None
    try:
        for step,(group,micros) in enumerate(micro_groups(lengths,0,physical)):
            if step>=steps+3: break
            optimizer.zero_grad(); torch.cuda.synchronize(); start=time.perf_counter()
            for indices in micros:
                micro_start=time.perf_counter(); batch=pad(tokenizer,encoded,indices,torch.device('cuda')); y=torch.tensor([labels[i] for i in indices],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16): loss=torch.nn.functional.cross_entropy(model(**batch).logits,y,reduction='sum')/len(group)
                scaler.scale(loss).backward(); torch.cuda.synchronize()
                if step>=3: micro_times.append(time.perf_counter()-micro_start); micro_count+=1
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
            if step>=3: measured.append(time.perf_counter()-start); examples+=len(group)
    except torch.cuda.OutOfMemoryError as exc: stable=False; error=str(exc)
    finally: stop.set(); monitor.join(timeout=2)
    elapsed=sum(measured)
    result={'physical_batch_size':physical,'gradient_accumulation':int(np.ceil(EFFECTIVE_BATCH/physical)),'effective_batch_size':EFFECTIVE_BATCH,'gradient_checkpointing':checkpointing,'mixed_precision':'fp16','stable':stable,'error':error,'optimizer_steps_measured':len(measured),'micro_batches_measured':micro_count,'samples_per_second':examples/elapsed if elapsed else 0,'batches_per_second':micro_count/elapsed if elapsed else 0,'seconds_per_micro_batch':sum(micro_times)/len(micro_times) if micro_times else None,'optimizer_updates_per_second':len(measured)/elapsed if elapsed else 0,'peak_cuda_allocated_gb':torch.cuda.max_memory_allocated()/1024**3,'peak_cuda_reserved_gb':torch.cuda.max_memory_reserved()/1024**3,'mean_gpu_utilization_percent':float(np.mean(utils)) if utils else None,'trainable_parameters':policy['trainable_parameters']}
    del model,optimizer,scaler; gc.collect(); torch.cuda.empty_cache()
    return result

if __name__=='__main__':
    require_reviewed_gate(explicit_user_approval=True)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    rows=load_rows()['train']; texts=[model_text(row) for row in rows]
    _,tokenizer,_=initialize_model(False); encoded=tokenizer(texts,padding=False,truncation=True,max_length=MAX_LENGTH); lengths=[len(x) for x in encoded['input_ids']]; labels=[row['label'] for row in rows]
    results=[run(encoded,lengths,labels,batch,True) for batch in [2,4,8]]
    safe=[r for r in results if r['stable'] and r['peak_cuda_reserved_gb']<=3.3]
    if not safe: raise RuntimeError('No stable safe benchmark configuration')
    selected=max(safe,key=lambda r:r['samples_per_second'])
    off=run(encoded,lengths,labels,selected['physical_batch_size'],False); results.append(off)
    if off['stable'] and off['peak_cuda_reserved_gb']<=3.3 and off['samples_per_second']>selected['samples_per_second']: selected=off
    output={'device':torch.cuda.get_device_name(0),'max_length':MAX_LENGTH,'selection_rule':'highest measured samples/sec among stable configurations with <=3.3 GiB reserved CUDA memory','results':results,'selected':selected}
    save_json(REPORTS/'metrics/gpu_benchmark.json',output); print(json.dumps(output,indent=2),flush=True)
