import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codelocator.config import MODELS,TRAINING
from codelocator.preprocessing import validate_processed
from codelocator.utils import save_json

def main():
    validate_processed()
    from huggingface_hub import HfApi,snapshot_download
    name=TRAINING['pretrained_model']; revision=HfApi().model_info(name).sha
    path=MODELS/'pretrained/codebert-base'
    snapshot_download(name,revision=revision,local_dir=path,allow_patterns=['config.json','tokenizer*','special_tokens_map.json','vocab.json','merges.txt','pytorch_model.bin','model.safetensors','README.md','LICENSE*'])
    save_json(path/'source_manifest.json',{'model':name,'revision':revision,'source':'https://huggingface.co/microsoft/codebert-base'})
    print('CodeBERT prepared at '+str(path),flush=True)
if __name__=='__main__': main()
