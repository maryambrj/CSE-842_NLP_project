from data_module import PeerDataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra 
import transformers
from transformers import Trainer
import os
#from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
import numpy as np
import random

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

    print("######################")
    print("Saving to: ", save_dir)
    print("######################")

    if os.path.exists(save_dir):
        print("Directory already exists")
        exit()

    max_length = 25000 #30000
    lr = 1e-5
    num_epochs = 2
    weight_decay = 0.01
    #eval_steps = max_steps + 100

    torch_format_dataset = PeerDataset(tokenizer=tokenizer, 
                                    max_length=max_length, 
                                    split='train')
    
    torch_format_dataset_test = PeerDataset(tokenizer=tokenizer, 
                                    max_length=max_length, 
                                    split='test')


    
    batch_size = 1
    gradient_accumulation_steps = 4
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"The length of dataset: {len(torch_format_dataset)},\nmax_steps: {max_steps},\nbatch_size: {batch_size},\naccumulation_step: {gradient_accumulation_steps}.")
    eval_steps = max_steps + 100


    warmup_steps = 2
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=5,
            logging_dir=f'{save_dir}/logs',
            output_dir=save_dir,
            optim="paged_adamw_32bit",
            save_steps=5, 
            ddp_find_unused_parameters= False,
            deepspeed='ds_config.json',
            weight_decay = weight_decay,
            evaluation_strategy = "steps",
            eval_steps = eval_steps,
            seed = seed,
    )
    
    #first get the base model architecture
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    model_path = 'llama_model'
    import re
    path_found = False
    for file in os.listdir(model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break


    if path_found:
        print("Loading from checkpoint")
        ## model_path is the place llama is saved to.
        model = AutoModelForCausalLM.from_pretrained(model_path, use_flash_attention_2="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
        
    else:
        print("Saving from HF and loading")
        ## model_id is the ID from HF. 
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2="true", torch_dtype=torch.bfloat16) # , device_map=device_map)
        model.save_pretrained(model_path)
        
    

    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset = torch_format_dataset,
        eval_dataset = torch_format_dataset_test,
        compute_metrics=None,         
        args=training_args,
        #seed = seed
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    #save the tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)




if __name__ == "__main__":
    main()
