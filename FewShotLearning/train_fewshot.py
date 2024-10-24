import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer
import openai
import pickle
import os
from datetime import datetime
import wandb

# find a safer way to store my api key here: ..
openai.api_key = os.getenv('API_KEY')

class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'labels': torch.tensor(self.data[idx]['labels'], dtype=torch.long),
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    # Decode the input_ids and labels to text
    input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    label_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
    return {
        'input_texts': input_texts,
        'label_texts': label_texts,
    }

def construct_prompt(input_text):
    prompt = ""
    for example in few_shot_examples:
        prompt += f"Text: {example['input_text']}\nSummary: {example['label_text']}\n\n"
    prompt += f"Text: {input_text}\nSummary:"
    return prompt

if __name__ == '__main__':

    # logging..
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"gpt3_fewshot_{current_time}"

    wandb.init(project='CSE-842_NLP_project', entity='maryam-brj', name=run_name)

    # Load the tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    print("Loading tokenized data...")
    with open('./processed_data/train_tokenized.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./processed_data/dev_tokenized.pkl', 'rb') as f:
        dev_data = pickle.load(f)

    print("Creating datasets and data loaders...")
    train_dataset = ReviewDataset(train_data)
    dev_dataset = ReviewDataset(dev_data)

    batch_size = 1  # GPT-3 allows 1 at a time: 
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    #few-shot examples: 
    few_shot_examples = [
        {
            'input_text': 'In this paper, we explore the effects of...', #we need to add the whole texts? perhaps import fromdifferent files..
            'label_text': 'This study investigates the impact of...'
        },
        {
            'input_text': 'The authors propose a novel method for...',
            'label_text': 'A new technique is introduced for...'
        },
    ]

    def generate_outputs(dataloader, output_file):
        all_outputs = []
        all_references = []

        for batch in dataloader:
            input_texts = batch['input_texts']
            label_texts = batch['label_texts']

            for input_text in input_texts:
                prompt = construct_prompt(input_text)
                # Call the OpenAI API
                response = openai.Completion.create(
                    engine='text-davinci-003',
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7,
                    n=1,
                    stop=None,
                )
                output_text = response.choices[0].text.strip()
                all_outputs.append(output_text)

                wandb.log({'input_text': input_text, 'output_text': output_text})

            all_references.extend(label_texts)

        with open(output_file, 'w', encoding='utf-8') as f:
            for output in all_outputs:
                f.write(output.strip() + '\n')

        if all_references:
            ref_file = output_file.replace('.txt', '_references.txt')
            with open(ref_file, 'w', encoding='utf-8') as f:
                for ref in all_references:
                    f.write(ref.strip() + '\n')

    print("Generating outputs for the dev set...")
    generate_outputs(dev_loader, './GPT3/dev_outputs.txt')

    print("Generation complete.")

    wandb.finish()
