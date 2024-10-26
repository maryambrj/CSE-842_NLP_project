import numpy as np
import random
import torch
from data_module import PeerDataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
import transformers
from transformers import Trainer
import os
#from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
import json

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    seed = 1001
    set_random_seed(seed)

    save_dir = 'trained_model'

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    #model_cfg = get_model_identifiers_from_yaml(cfg.model_family)

    #model_id = "NousResearch/Llama-2-7b-chat-hf"
    model_id = "togethercomputer/LLaMA-2-7B-32K"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token


    data_path = f'extracted_data/train_data.json'
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    paper_tokenizer_len = []
    review_tokenizer_len = []
    
    del data[4714]
    del data[4714]
    del data[4714]
    #import pdb;pdb.set_trace()

    for idx, pr_pair in enumerate(data):
        #if idx == 4714 or idx == 4715 or idx == 4716:
        #continue
        
        print(idx)
        paper = pr_pair['input']
        review = pr_pair['output']

        paper_start_token, paper_end_token, review_token = "[INST] ", " [/INST]", ""
        new_paper = paper_start_token + paper + paper_end_token
        new_review = review_token + review

        num_paper_tokens = len(tokenizer.tokenize(new_paper, add_special_tokens=True))
        num_review_tokens = len(tokenizer.tokenize(new_review, add_special_tokens=True))

        paper_tokenizer_len.append(num_paper_tokens)
        review_tokenizer_len.append(num_review_tokens)

    
    paper_token_max = np.max(np.array(paper_tokenizer_len))
    paper_token_std = np.std(np.array(paper_tokenizer_len))

    review_token_max = np.max(np.array(review_tokenizer_len))
    review_token_std = np.std(np.array(review_tokenizer_len))
    
    print(paper_token_max)
    #print(paper_token_std)

    print(review_token_max)
    #print(review_token_std)




if __name__ == "__main__":
    main()
