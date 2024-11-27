import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig
import pickle
from torch.nn.utils.rnn import pad_sequence
import os

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

    # Dynamically pad the input_ids and labels
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Create attention masks
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

if __name__ == '__main__':
    model_dir = './BART/best_model'
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Ensure that `decoder_start_token_id` is set
    generation_config = GenerationConfig(
        early_stopping=True,
        num_beams=4,
        no_repeat_ngram_size=5, #3,
        max_length=512,  # Adjust max length as needed
        decoder_start_token_id=tokenizer.bos_token_id,  # Explicitly set the decoder start token ID
    )


    print("Loading tokenized data...")
    with open('./processed_data/dev_tokenized.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open('./processed_data/test_tokenized.pkl', 'rb') as f:
        test_data = pickle.load(f)

    dev_dataset = ReviewDataset(dev_data)
    test_dataset = ReviewDataset(test_data)

    dev_loader = DataLoader(dev_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)


    def generate_outputs(dataloader, output_file):
        all_outputs = []  # Reset after each dataset
        all_references = []  # Reset references as well
    
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
    
                # Generate outputs
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=generation_config.num_beams,
                    no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
                    max_length=generation_config.max_length,
                    early_stopping=generation_config.early_stopping,
                    decoder_start_token_id=generation_config.decoder_start_token_id,
                )
    
                # Decode outputs and append to the final list
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_outputs.extend(decoded_outputs)
    
    
                # Decode labels (optional, for evaluation)
                if 'labels' in batch:
                    decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                    all_references.extend(decoded_labels)
    
        # Write generated outputs to file (one output per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for output in all_outputs:
                f.write(output.strip() + '\n')

        # Write references to a separate file (for evaluation purposes)
        if all_references:
            ref_file = output_file.replace('.txt', '_references.txt')
            with open(ref_file, 'w', encoding='utf-8') as f:
                for ref in all_references:
                    f.write(ref.strip() + '\n')




    print("Generating outputs for the dev set...")
    generate_outputs(dev_loader, './BART/dev_outputs.txt')

    print("Generating outputs for the test set...")
    generate_outputs(test_loader, './BART/test_outputs.txt')

    print("Generation complete.")
