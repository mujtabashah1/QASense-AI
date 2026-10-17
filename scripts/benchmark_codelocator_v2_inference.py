import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from codelocator.utils import save_json
from codelocator_v2.config import REPORTS
from codelocator_v2.context import model_text
from codelocator_v2.runtime import MAX_LENGTH,initialize_model,load_rows,predict

if __name__=='__main__':
    import torch
    path=REPORTS/'metrics/gpu_benchmark.json'; benchmark=json.loads(path.read_text()); selected=benchmark['selected']; physical=selected['physical_batch_size']
    rows=load_rows()['validation']; model,tokenizer,_=initialize_model(False); model.cuda()
    encoded=tokenizer([model_text(row) for row in rows],padding=False,truncation=True,max_length=MAX_LENGTH); lengths=[len(x) for x in encoded['input_ids']]
    ordered=sorted(range(len(rows)),key=lambda i:lengths[i]); take=np.linspace(0,len(ordered)-1,min(2048,len(ordered)),dtype=int); indices=[ordered[i] for i in take]
    subset={key:[values[i] for i in indices] for key,values in encoded.items()}; subset_lengths=[lengths[i] for i in indices]
    predict(model,tokenizer,{key:values[:min(64,len(values))] for key,values in subset.items()},subset_lengths[:min(64,len(subset_lengths))],physical,torch.device('cuda'))
    torch.cuda.synchronize(); start=time.perf_counter(); predict(model,tokenizer,subset,subset_lengths,physical,torch.device('cuda')); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    rate=len(indices)/elapsed; train_count=38810; validation_count=15975; test_count=14381
    benchmark['inference_benchmark']={'representative_samples':len(indices),'seconds':elapsed,'samples_per_second':rate,'physical_batch_size':physical}
    train_epoch=train_count/selected['samples_per_second']; validation=validation_count/rate
    benchmark['estimated_seconds']={'training_per_epoch':train_epoch,'validation_per_epoch':validation,'three_epochs_with_validation':3*(train_epoch+validation),'final_validation_and_test':(validation_count+test_count)/rate,'maximum_total':3*(train_epoch+validation)+(validation_count+test_count)/rate}
    save_json(path,benchmark); print(json.dumps(benchmark['inference_benchmark']|{'estimated_seconds':benchmark['estimated_seconds']},indent=2))
