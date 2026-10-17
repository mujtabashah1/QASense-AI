from .config import PROCESSED, MODELS, REPORTS
from .context_builder import format_candidate
import json
from pathlib import Path
from .utils import read_jsonl,save_json,file_digest
from .train_bug_type import fit_with_oom_recovery,predict_transformer
from .evaluate import ranking_metrics,classification_metrics

def _evaluate_splits(model,tokenizer,cfg,device,splits):
    from collections import defaultdict
    result={}
    for name in ['validation','test']:
        rows=splits[name]
        texts=[format_candidate(r['candidate_line'],r['context']) for r in rows]
        labels=[r['label'] for r in rows]
        probabilities=predict_transformer(model,tokenizer,texts,cfg,device)
        result[name]={'ranking':ranking_metrics(rows,probabilities[:,1]),'candidate_classification':classification_metrics(labels,probabilities,['CONTEXT_LINE','ANNOTATED_BUG_LINE'])}
        groups=defaultdict(list)
        for row in rows: groups[row['sample_id']].append(row['line_number'])
        position_scores=[-abs(row['line_number']-(min(groups[row['sample_id']])+max(groups[row['sample_id']]))/2) for row in rows]
        result[name]['position_only_diagnostic']=ranking_metrics(rows,position_scores)
        save_json(REPORTS/'metrics/localization_evaluation_progress.json',{'completed_splits':list(result),'results':result})
    return result

def evaluate_existing_localizer(metadata):
    """Evaluate the persisted best checkpoint without performing any training."""
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    model_dir=MODELS/'line_localizer'
    required=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','training_metadata.json']
    missing=[name for name in required if not (model_dir/name).is_file()]
    if missing: raise FileNotFoundError(f'Incomplete saved localizer checkpoint: {missing}')
    training_metadata=json.loads((model_dir/'training_metadata.json').read_text(encoding='utf-8'))
    if training_metadata.get('task')!='localizer' or training_metadata.get('selection_metric')!='MRR':
        raise ValueError('Saved checkpoint metadata is not the validation-MRR-selected localizer')
    if not metadata['context']['localization_scientifically_supported']:
        raise ValueError('Processed metadata does not support localization evaluation')
    splits={name:read_jsonl(PROCESSED/f'splits/{name}_localization.jsonl') for name in ['validation','test']}
    if any(len(rows)<20 or not any(row['label'] for row in rows) for rows in splits.values()):
        raise ValueError('Not enough verified localization examples in every evaluation split')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AutoModelForSequenceClassification.from_pretrained(model_dir,local_files_only=True).to(device)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    result=_evaluate_splits(model,tokenizer,training_metadata['config'],device,splits)
    save_json(REPORTS/'metrics/localization.json',result)
    save_json(model_dir/'evaluation.json',result)
    save_json(REPORTS/'metrics/localizer_recovery.json',{
        'recovery_mode':'evaluate_saved_best_checkpoint_only',
        'training_performed':False,
        'checkpoint_sha256':file_digest(model_dir/'model.safetensors'),
        'selected_after_completed_epoch':2,
        'selection_metric':'validation MRR',
        'best_validation_score_at_save':training_metadata['best_validation_score'],
        'evaluation_splits':{name:{'candidate_lines':len(rows),'sha256':file_digest(PROCESSED/f'splits/{name}_localization.jsonl')} for name,rows in splits.items()}
    })
    del model
    if device.type=='cuda': torch.cuda.empty_cache()
    return result

def train_localizer(metadata):
    if not metadata['context']['localization_scientifically_supported']:
        save_json(REPORTS/'metrics/localization_status.json',{'status':'unsupported','reason':'Not enough verified genuine context; no fabricated localization data.'})
        return None
    splits={n:read_jsonl(PROCESSED/f'splits/{n}_localization.jsonl') for n in ['train','validation','test']}
    if any(len(rows)<20 or not any(r['label'] for r in rows) for rows in splits.values()): raise ValueError('Not enough verified localization examples in every split')
    texts=lambda n:[format_candidate(r['candidate_line'],r['context']) for r in splits[n]]
    ys=lambda n:[r['label'] for r in splits[n]]
    model,tokenizer,cfg,device=fit_with_oom_recovery(texts('train'),ys('train'),texts('validation'),ys('validation'),['CONTEXT_LINE','ANNOTATED_BUG_LINE'],MODELS/'line_localizer','localizer',selection_function=lambda probabilities:ranking_metrics(splits['validation'],probabilities[:,1])['mrr'])
    result=_evaluate_splits(model,tokenizer,cfg,device,splits)
    save_json(REPORTS/'metrics/localization.json',result)
    save_json(MODELS/'line_localizer/evaluation.json',result)
    return result
