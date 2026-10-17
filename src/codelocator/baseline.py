import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .config import MODELS, REPORTS, SEED
from .evaluate import classification_metrics
from .utils import save_json,digest

def training_signature(records):
    import json
    return digest(json.dumps([(r['buggy_statement'],r['bug_type']) for r in records],ensure_ascii=False))

def train_baseline(splits,mapping):
    model=Pipeline([('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=50000,dtype=np.float32,lowercase=False)),('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED,C=1.0))])
    model.fit([r['buggy_statement'] for r in splits['train']],[r['bug_type'] for r in splits['train']])
    names=sorted(mapping['label2id'],key=mapping['label2id'].get)
    results={}
    for split in ['validation','test']:
        probabilities=model.predict_proba([r['buggy_statement'] for r in splits[split]])
        indices=[list(model.named_steps['clf'].classes_).index(label) for label in names]
        probabilities=probabilities[:,indices]
        results[split]=classification_metrics([mapping['label2id'][r['bug_type']] for r in splits[split]],probabilities,names)
    joblib.dump(model,MODELS/'baseline/bug_type_tfidf.joblib')
    save_json(MODELS/'baseline/metadata.json',{'model':'char TF-IDF + balanced logistic regression','input':'buggy_statement only','feature_policy':{'analyzer':'char','ngram_range':[3,5],'lowercase':False},'labels':mapping,'training_samples':len(splits['train']),'training_signature':training_signature(splits['train']),'seed':SEED,'metrics':results})
    save_json(REPORTS/'metrics/baseline.json',results)
    print('Baseline complete: '+str({k:{m:v[m] for m in ['accuracy','macro_f1','weighted_f1']} for k,v in results.items()}),flush=True)
    return results
