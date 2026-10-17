import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def classification_metrics(labels,probabilities,names):
    labels=np.asarray(labels); probabilities=np.asarray(probabilities)
    prediction=probabilities.argmax(axis=1)
    precision,recall,f1,_=precision_recall_fscore_support(labels,prediction,labels=list(range(len(names))),average='macro',zero_division=0)
    weighted=precision_recall_fscore_support(labels,prediction,labels=list(range(len(names))),average='weighted',zero_division=0)[2]
    top=np.argsort(-probabilities,axis=1,kind='stable')[:,:min(3,len(names))]
    return {'accuracy':float(accuracy_score(labels,prediction)),'top_1_accuracy':float(accuracy_score(labels,prediction)),'macro_precision':float(precision),'macro_recall':float(recall),'macro_f1':float(f1),'weighted_f1':float(weighted),'top_3_accuracy':float(np.mean([label in row for label,row in zip(labels,top)])),'per_class':classification_report(labels,prediction,labels=list(range(len(names))),target_names=names,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(labels,prediction,labels=list(range(len(names)))).tolist()}

def ranking_metrics(candidates,scores):
    from collections import defaultdict
    groups=defaultdict(list)
    for c,s in zip(candidates,scores): groups[c['sample_id']].append((float(s),c['line_number'],c['label']))
    ranks=[]
    for rows in groups.values():
        rows.sort(key=lambda r:(-r[0],r[1]))
        positive=[i for i,r in enumerate(rows,1) if r[2]]
        if len(positive)!=1: raise ValueError('Each localization sample must have exactly one labelled line')
        ranks.append(positive[0])
    if not ranks: raise ValueError('No localization samples')
    return {'samples':len(ranks),'top_1':float(np.mean(np.array(ranks)<=1)),'top_3':float(np.mean(np.array(ranks)<=3)),'top_5':float(np.mean(np.array(ranks)<=5)),'mrr':float(np.mean(1/np.array(ranks))),'mean_rank':float(np.mean(ranks)),'median_rank':float(np.median(ranks))}
