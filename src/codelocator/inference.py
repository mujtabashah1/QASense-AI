import json
from pathlib import Path
import joblib
import numpy as np
from .config import MODELS,TRAINING
from .context_builder import extract_context,format_candidate

class CodeLocator:
    def __init__(self,backend='auto'):
        if backend not in ['auto','baseline','codebert']: raise ValueError('backend must be auto, baseline or codebert')
        selection=MODELS/'selection.json'
        if backend=='auto': backend=json.loads(selection.read_text()).get('bug_type_backend','baseline') if selection.exists() else 'baseline'
        self.backend=backend; self.classifier=None; self.tokenizer=None; self.localizer=None; self.localizer_tokenizer=None
        if backend=='baseline':
            path=MODELS/'baseline/bug_type_tfidf.joblib'
            if not path.is_file(): raise FileNotFoundError('CodeLocator baseline not trained')
            self.baseline=joblib.load(path)
        else:
            self.classifier,self.tokenizer,self.device,self.classifier_config=self.load_transformer(MODELS/'bug_type_classifier')
        if (MODELS/'line_localizer/training_metadata.json').exists() and (MODELS/'line_localizer/evaluation.json').exists():
            self.localizer,self.localizer_tokenizer,self.device,self.localizer_config=self.load_transformer(MODELS/'line_localizer')

    @staticmethod
    def load_transformer(path):
        if not (Path(path)/'training_metadata.json').is_file(): raise FileNotFoundError(f'Trained transformer not found: {path}')
        import torch
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metadata=json.loads((Path(path)/'training_metadata.json').read_text())
        return AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(device),AutoTokenizer.from_pretrained(path,local_files_only=True),device,metadata['config']

    def analyze_code(self,code):
        if not isinstance(code,str) or not code.strip(): raise ValueError('code must be a nonempty Java source string')
        if '\x00' in code: raise ValueError('NUL characters are not valid source input')
        if len(code)>1_000_000: raise ValueError('Input exceeds one million characters; analyze smaller files')
        normalized=code.replace('\r\n','\n').replace('\r','\n')
        if self.backend=='baseline':
            probability=self.baseline.predict_proba([normalized])[0]; labels=list(self.baseline.named_steps['clf'].classes_)
        else:
            from .train_bug_type import predict_transformer
            probability=predict_transformer(self.classifier,self.tokenizer,[normalized],self.classifier_config,self.device)[0]
            labels=[self.classifier.config.id2label[i] for i in range(len(probability))]
        best=int(np.argmax(probability)); suspicious=[]
        if self.localizer is not None:
            from .train_bug_type import predict_transformer
            lines=normalized.split('\n'); candidates=[]
            for i,line in enumerate(lines):
                if line.strip(): candidates.append({'line_number':i+1,'code':line,'text':format_candidate(line,extract_context(lines,i,TRAINING['candidate_radius'],TRAINING['candidate_radius'])['context'])})
            scores=predict_transformer(self.localizer,self.localizer_tokenizer,[r['text'] for r in candidates],self.localizer_config,self.device)[:,1]
            suspicious=sorted([{'line_number':r['line_number'],'score':float(s),'code':r['code']} for r,s in zip(candidates,scores)],key=lambda r:(-r['score'],r['line_number']))[:5]
        return {'prediction':'Defect presence not assessed','bug_probability':None,'predicted_bug_type':labels[best],'bug_type_confidence':float(probability[best]),'bug_type_backend':self.backend,'suspicious_lines':suspicious,'localization_available':self.localizer is not None,'limitations':['Bug category is conditional on buggy Java fragments; confidence is not an overall defect probability.','Full-file bug category classification is outside the statement training distribution.','Line scores, when available, rank suspiciousness and do not prove exact faulty locations.']}

def analyze_code(code,backend='auto'): return CodeLocator(backend).analyze_code(code)
