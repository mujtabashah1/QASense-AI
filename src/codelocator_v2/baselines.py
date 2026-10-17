import random
import numpy as np
from codelocator.evaluate import ranking_metrics
from .config import SEED

def position_baseline(rows):
    scores=[-abs(r['line_number']-(r['region_start_line']+r['region_end_line'])/2) for r in rows]
    return ranking_metrics(rows,scores)

def random_baseline(rows,seeds=range(SEED,SEED+5)):
    runs=[]
    for seed in seeds:
        rng=random.Random(seed); runs.append(ranking_metrics(rows,[rng.random() for _ in rows]))
    keys=runs[0]
    return {'seeds':list(seeds),'runs':runs,'mean':{k:float(np.mean([r[k] for r in runs])) for k in keys},'std':{k:float(np.std([r[k] for r in runs])) for k in keys}}
