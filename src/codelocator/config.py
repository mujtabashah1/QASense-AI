from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/'datasets/manysstubs4j'
PROCESSED = ROOT/'datasets/processed/codelocator'
REPORTS = ROOT/'reports/codelocator'
MODELS = ROOT/'models/codelocator'
SEED = 42
SCHEMA = {'project':'projectName','commit':'fixCommitSHA1','parent_commit':'fixCommitParentSHA1','file_path':'bugFilePath','bug_type':'bugType','buggy_statement':'sourceBeforeFix','fixed_statement':'sourceAfterFix','buggy_line_number':'bugLineNum','patch':'fixPatch'}
TRAINING = {'pretrained_model':'microsoft/codebert-base','max_length':256,'batch_size':2,'gradient_accumulation':8,'epochs':3,'learning_rate':2e-5,'weight_decay':.01,'warmup_ratio':.1,'early_stopping_patience':2,'candidate_radius':3,'source_radius':15,'negative_lines_per_sample':6}
TRAINING['train_final_encoder_layers'] = 4
