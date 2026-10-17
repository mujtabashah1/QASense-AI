import json,math,random,time
from pathlib import Path
import numpy as np
from codelocator.utils import save_json
from .config import MODELS,PROCESSED,REPORTS,SEED
from .context import model_text
from .gate import require_reviewed_gate
from .runtime import EFFECTIVE_BATCH,MAX_LENGTH,atomic_torch_save,evaluation,initialize_model,load_rows,log,micro_groups,pad,predict,restore_rng,rng_state,split_identity

CHECKPOINT_EVERY=500

def _checkpoint_state(model,optimizer,scheduler,scaler,config,identity,epoch,next_group,phase,global_step,total_loss,samples,history,best,stale):
    return {'schema_version':1,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),'epoch':epoch,'next_group_index':next_group,'phase':phase,'global_optimizer_step':global_step,'epoch_training_loss_sum':total_loss,'epoch_samples_processed':samples,'history':history,'best_validation_mrr':best,'stale_epochs':stale,'config':config,'split_identity':identity,'rng_state':rng_state()}

def _save_checkpoint(state,epoch_snapshot=None):
    checkpoint_dir=MODELS/'checkpoints'
    if epoch_snapshot is not None: atomic_torch_save(checkpoint_dir/f'epoch_{epoch_snapshot}/training_state.pt',state)
    atomic_torch_save(checkpoint_dir/'latest.pt',state)

def _status(**values): save_json(REPORTS/'metrics/training_status.json',values)

def train_v2():
    require_reviewed_gate(explicit_user_approval=True)
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    identity=split_identity(); benchmark=json.loads((REPORTS/'metrics/gpu_benchmark.json').read_text()); selected=benchmark['selected']
    config={'physical_batch_size':selected['physical_batch_size'],'gradient_accumulation':selected['gradient_accumulation'],'effective_batch_size':EFFECTIVE_BATCH,'gradient_checkpointing':selected['gradient_checkpointing'],'mixed_precision':'fp16','max_length':MAX_LENGTH,'epochs':3,'learning_rate':2e-5,'weight_decay':.01,'warmup_ratio':.1,'early_stopping_patience':2,'initialization':'models/codelocator/line_localizer (v1 epoch-2 validation-MRR-selected weights)','checkpoint_every_optimizer_steps':CHECKPOINT_EVERY}
    rows=load_rows(); train_rows=rows['train']; val_rows=rows['validation']
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model,tokenizer,policy=initialize_model(config['gradient_checkpointing']); device=torch.device('cuda'); model.to(device)
    train_encoded=tokenizer([model_text(r) for r in train_rows],padding=False,truncation=True,max_length=MAX_LENGTH); train_lengths=[len(x) for x in train_encoded['input_ids']]
    val_encoded=tokenizer([model_text(r) for r in val_rows],padding=False,truncation=True,max_length=MAX_LENGTH); val_lengths=[len(x) for x in val_encoded['input_ids']]
    counts=np.bincount([r['label'] for r in train_rows],minlength=2); weights=np.sqrt(counts.sum()/np.maximum(counts,1)); weights/=weights.mean(); loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),reduction='sum')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    steps_per_epoch=math.ceil(len(train_rows)/EFFECTIVE_BATCH); total_steps=steps_per_epoch*config['epochs']; warmup=max(1,int(total_steps*config['warmup_ratio']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warmup,max(0,(total_steps-step)/max(1,total_steps-warmup))))
    scaler=torch.amp.GradScaler('cuda'); epoch=0; next_group=0; phase='training'; global_step=0; total_loss=0.; samples=0; history=[]; best=-1.; stale=0
    latest=MODELS/'checkpoints/latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['config']!=config or state['split_identity']!=identity: raise ValueError('Resume checkpoint configuration or split identity mismatch')
        model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer_state_dict']); scheduler.load_state_dict(state['scheduler_state_dict']); scaler.load_state_dict(state['scaler_state_dict'])
        epoch=state['epoch']; next_group=state['next_group_index']; phase=state['phase']; global_step=state['global_optimizer_step']; total_loss=state['epoch_training_loss_sum']; samples=state['epoch_samples_processed']; history=state['history']; best=state['best_validation_mrr']; stale=state['stale_epochs']; restore_rng(state['rng_state'])
        log(f'RESUMED epoch={epoch+1} phase={phase} next_group={next_group} global_step={global_step}')
    else: log('NEW RUN initialized from protected v1 epoch-2 weights; this is not an optimizer-level continuation of v1 epoch 3')
    run_start=time.time()
    while epoch<config['epochs'] and stale<config['early_stopping_patience']:
        groups=list(micro_groups(train_lengths,epoch,config['physical_batch_size']))
        micro_prefix=[]; micro_total=0
        for _,microbatches in groups:
            micro_total+=len(microbatches); micro_prefix.append(micro_total)
        if phase=='training':
            model.train(); epoch_start=time.time()
            for group_index in range(next_group,len(groups)):
                group,micros=groups[group_index]; optimizer.zero_grad(); denominator=float(sum(weights[train_rows[i]['label']] for i in group)); group_loss=0.
                for indices in micros:
                    batch=pad(tokenizer,train_encoded,indices,device); labels=torch.tensor([train_rows[i]['label'] for i in indices],device=device)
                    with torch.autocast('cuda',dtype=torch.float16): logits=model(**batch).logits; loss=loss_fn(logits.float(),labels)/denominator
                    group_loss+=loss.item(); scaler.scale(loss).backward()
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
                total_loss+=group_loss; samples+=len(group); global_step+=1; completed=group_index+1
                elapsed=time.time()-epoch_start; remaining=elapsed/max(1,completed-next_group)*(len(groups)-completed)
                if completed%50==0:
                    _status(status='training',epoch=epoch+1,max_epochs=config['epochs'],micro_batch=micro_prefix[group_index],micro_batches_in_epoch=micro_total,optimizer_step=completed,optimizer_steps_in_epoch=len(groups),global_optimizer_step=global_step,best_validation_mrr=None if best<0 else best,samples_processed=samples,elapsed_seconds=time.time()-run_start,estimated_epoch_seconds_remaining=remaining,physical_batch_size=config['physical_batch_size'],gradient_accumulation=config['gradient_accumulation'])
                    log(f'epoch={epoch+1} step={completed}/{len(groups)} loss={total_loss/completed:.6f} lr={scheduler.get_last_lr()[0]:.8g} eta_seconds={remaining:.0f}')
                if completed%CHECKPOINT_EVERY==0:
                    state=_checkpoint_state(model,optimizer,scheduler,scaler,config,identity,epoch,completed,'training',global_step,total_loss,samples,history,best,stale); _save_checkpoint(state); log(f'checkpoint saved epoch={epoch+1} next_group={completed}')
            phase='validation'; next_group=len(groups)
            state=_checkpoint_state(model,optimizer,scheduler,scaler,config,identity,epoch,next_group,phase,global_step,total_loss,samples,history,best,stale); _save_checkpoint(state); log(f'validation start epoch={epoch+1}')
        _status(status='validating',epoch=epoch+1,max_epochs=config['epochs'],global_optimizer_step=global_step,best_validation_mrr=None if best<0 else best,elapsed_seconds=time.time()-run_start)
        validation_start=time.time(); probabilities=predict(model,tokenizer,val_encoded,val_lengths,config['physical_batch_size'],device); metrics=evaluation(val_rows,probabilities); score=metrics['ranking']['mrr']; row={'epoch':epoch+1,'training_loss':total_loss/len(groups),'validation':metrics,'validation_seconds':time.time()-validation_start}; history.append(row); save_json(REPORTS/'metrics/training_history.json',history)
        improved=score>best
        if improved:
            best=score; stale=0; output=MODELS/'line_localizer'; model.save_pretrained(output,safe_serialization=True); tokenizer.save_pretrained(output)
            save_json(output/'training_metadata.json',{'task':'codelocator_v2_localizer','config':config,'layer_policy':policy,'best_epoch':epoch+1,'best_validation_mrr':best,'selection_metric':'debiased_validation_mrr','split_identity':identity,'initialization_statement':'CodeLocator v2 was initialized from the v1 epoch-2 model weights. This is a new v2 fine-tuning run on position-debiased localization data and is NOT an optimizer-level continuation of the interrupted v1 epoch 3.'})
        else: stale+=1
        log(f'validation end epoch={epoch+1} mrr={score:.6f} top1={metrics["ranking"]["top_1"]:.6f} top3={metrics["ranking"]["top_3"]:.6f} top5={metrics["ranking"]["top_5"]:.6f} improved={improved} stale={stale}')
        epoch+=1; next_group=0; phase='training'; total_loss=0.; samples=0
        state=_checkpoint_state(model,optimizer,scheduler,scaler,config,identity,epoch,next_group,phase,global_step,total_loss,samples,history,best,stale); _save_checkpoint(state,epoch_snapshot=epoch); log(f'completed epoch checkpoint saved epoch={epoch}')
    _status(status='evaluating_best_checkpoint',completed_epochs=epoch,best_validation_mrr=best,global_optimizer_step=global_step,elapsed_seconds=time.time()-run_start)
    del model; torch.cuda.empty_cache(); model=AutoModelForSequenceClassification.from_pretrained(MODELS/'line_localizer',local_files_only=True).to(device); tokenizer=AutoTokenizer.from_pretrained(MODELS/'line_localizer',local_files_only=True)
    results={}
    for split in ['validation','test']:
        encoded=tokenizer([model_text(r) for r in rows[split]],padding=False,truncation=True,max_length=MAX_LENGTH); lengths=[len(x) for x in encoded['input_ids']]; results[split]=evaluation(rows[split],predict(model,tokenizer,encoded,lengths,config['physical_batch_size'],device))
    natural_ids={r['sample_id'] for r in rows['test'] if r['region_kind'] in {'method','block'}}; natural=[r for r in rows['test'] if r['sample_id'] in natural_ids]
    if natural:
        encoded=tokenizer([model_text(r) for r in natural],padding=False,truncation=True,max_length=MAX_LENGTH); lengths=[len(x) for x in encoded['input_ids']]; results['natural_context_test']=evaluation(natural,predict(model,tokenizer,encoded,lengths,config['physical_batch_size'],device))
    save_json(REPORTS/'metrics/evaluation.json',results); save_json(MODELS/'line_localizer/evaluation.json',results)
    success=results['test']['ranking']['mrr']>json.loads((REPORTS/'metrics/position_baseline.json').read_text())['test']['mrr']
    _status(status='complete',completed_epochs=epoch,early_stopping_criterion_reached=stale>=config['early_stopping_patience'],stopped_before_max_epochs=epoch<config['epochs'],best_validation_mrr=best,global_optimizer_step=global_step,total_elapsed_seconds=time.time()-run_start,beats_position_only_test_mrr=success)
    log(f'COMPLETE epochs={epoch} best_validation_mrr={best:.6f} test_mrr={results["test"]["ranking"]["mrr"]:.6f} beats_position={success}')
    return results
