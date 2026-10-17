import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, accuracy_score, brier_score_loss, log_loss

def evaluate(y, probability, threshold=.5):
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0,1]).ravel()
    return {'precision': precision_score(y,prediction,zero_division=0), 'recall': recall_score(y,prediction,zero_division=0), 'f1': f1_score(y,prediction,zero_division=0), 'roc_auc': roc_auc_score(y,probability) if len(np.unique(y))==2 else None, 'pr_auc': average_precision_score(y,probability), 'balanced_accuracy': balanced_accuracy_score(y,prediction), 'mcc': matthews_corrcoef(y,prediction), 'specificity': float(tn/(tn+fp)) if tn+fp else None, 'accuracy': accuracy_score(y,prediction), 'confusion_matrix': [[int(tn),int(fp)],[int(fn),int(tp)]], 'brier': brier_score_loss(y,probability), 'log_loss': log_loss(y,probability,labels=[0,1])}
