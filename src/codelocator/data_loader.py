from pathlib import Path
import pandas as pd
from .config import RAW, SCHEMA

def load_parquet(path):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f'Dataset not found: {path}')
    try: frame=pd.read_parquet(path)
    except (ImportError,OSError) as e:
        raise RuntimeError('Parquet backend failed. Do not downgrade packages repeatedly; use the supplied Linux/WSL workflow. '+str(e)) from e
    if frame.empty: raise ValueError(f'Dataset is empty: {path}')
    return frame

def load_primary():
    frame=load_parquet(RAW/'sstubsLarge/train-00000-of-00001.parquet')
    missing=set(SCHEMA.values())-set(frame.columns)
    if missing: raise ValueError(f'Discovered schema has changed; missing fields: {missing}')
    return frame
