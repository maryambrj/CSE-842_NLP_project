import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
import json

def convert_raw_data_to_model_format(tokenizer, max_length,  paper, review):

    paper_start_token, paper_end_token, review_token = "[INST] ", " [/INST]", ""
    new_paper = paper_start_token + paper + paper_end_token
    new_review = review_token + review
    full_text = new_paper + new_review
    num_paper_tokens = len(tokenizer.tokenize(new_paper, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    
    #print(num_paper_tokens)
    #import pdb;pdb.set_trace()
    #change label to -100 for paper tokens
    for i in range(num_paper_tokens): 
        #print(i)
        label[i] = -100

    #encoded_padded_data = {}
    encoded['input_ids'] = torch.tensor(pad_input_ids)
    encoded['attention_mask'] = torch.tensor(pad_attention_mask)
    encoded['labels'] = torch.tensor(label)

    return encoded

    #return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)



class PeerDataset(Dataset):
    def __init__(self, tokenizer, max_length=2048, split='train'):
        super(PeerDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert split=='train' or split=='test'
        data_path = f'extracted_data/{split}_data.json'

        with open(data_path, 'r') as file:
            self.data = json.load(file)
        
        if split == 'train':
            del self.data[4714]
            del self.data[4714]
            del self.data[4714]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pr_pair = self.data[idx]
        paper = pr_pair['input']
        review = pr_pair['output']

        converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, paper, review)
            
        return converted_data



def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

