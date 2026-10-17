from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'datasets/manysstubs4j'
V1_PROCESSED=ROOT/'datasets/processed/codelocator'
PROCESSED=ROOT/'datasets/processed/codelocator_v2'
REPORTS=ROOT/'reports/codelocator_v2'
MODELS=ROOT/'models/codelocator_v2'
SEED=42
MAX_REGION_LINES=61
TRAIN_NEGATIVES=6
POSITION_MRR_LIMIT=.75
MIDDLE_BUCKET_LIMIT=.40
SINGLE_BUCKET_LIMIT=.45
