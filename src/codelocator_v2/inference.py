import hashlib,json
from .config import MODELS
from .context import eligible,model_text,natural_region
from .runtime import MAX_LENGTH,predict

class CodeLocatorV2:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        path=MODELS/'line_localizer'
        if not (path/'evaluation.json').is_file(): raise FileNotFoundError('Evaluated CodeLocator v2 localizer not found')
        self.metadata=json.loads((path/'training_metadata.json').read_text(encoding='utf-8'))
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(self.device)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)

    def locate(self,code,top_k=5):
        if not isinstance(code,str) or not code.strip(): raise ValueError('code must be nonempty Java source')
        if '\x00' in code or len(code)>1_000_000: raise ValueError('invalid or oversized source')
        normalized=code.replace('\r\n','\n').replace('\r','\n'); lines=normalized.split('\n'); source_id=hashlib.sha256(normalized.encode()).hexdigest(); rows=[]
        for index,line in enumerate(lines):
            if not eligible(line): continue
            region=natural_region(lines,index,source_id+':'+str(index))
            rows.append({'sample_id':source_id,'line_number':index+1,'candidate_line':line,'context':'\n'.join(region['lines']),'label':0,'region_start_line':region['start']+1,'region_end_line':region['end']+1,'region_kind':region['kind']})
        if not rows: return {'localizer_version':'v2','suspicious_lines':[],'limitation':'Scores rank suspiciousness and do not prove an exact defect.'}
        encoded=self.tokenizer([model_text(row) for row in rows],padding=False,truncation=True,max_length=MAX_LENGTH); lengths=[len(x) for x in encoded['input_ids']]
        scores=predict(self.model,self.tokenizer,encoded,lengths,self.metadata['config']['physical_batch_size'],self.device)[:,1]
        ranked=sorted(({'line_number':row['line_number'],'score':float(score),'code':row['candidate_line']} for row,score in zip(rows,scores)),key=lambda row:(-row['score'],row['line_number']))[:top_k]
        return {'localizer_version':'v2','suspicious_lines':ranked,'limitation':'Scores rank suspiciousness and do not prove an exact defect.'}

def locate_code(code,top_k=5): return CodeLocatorV2().locate(code,top_k)
