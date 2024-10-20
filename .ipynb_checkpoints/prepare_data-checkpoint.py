import os
import json
import re
import torch
from transformers import BartTokenizer

# Define helper functions
def extract_input_output_pairs(data_split_path):
    reviews_path = os.path.join(data_split_path, 'reviews')
    parsed_pdfs_path = os.path.join(data_split_path, 'parsed_pdfs')
    data_pairs = []

    for review_file in os.listdir(reviews_path):
        if review_file.endswith('.json'):
            review_file_path = os.path.join(reviews_path, review_file)
            with open(review_file_path, 'r') as f:
                paper_data = json.load(f)

            # Extract paper content
            paper_id = paper_data.get('paper_id', '')
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            full_body = ''

            # Load the parsed PDF content (if needed)
            parsed_pdf_file = os.path.join(parsed_pdfs_path, f"{paper_id}.pdf.json")
            if os.path.exists(parsed_pdf_file):
                with open(parsed_pdf_file, 'r') as f_pdf:
                    parsed_pdf = json.load(f_pdf)
                    full_body = ' '.join([section.get('text', '') for section in parsed_pdf.get('sections', [])])

            # Choose whether to use abstract, full body, or both
            paper_content = abstract  # or full_body, or abstract + full_body

            # Extract reviews
            reviews = paper_data.get('reviews', [])
            for review in reviews:
                # Concatenate review comments and summary
                review_text = ''
                comments = review.get('comments', '')
                summary = review.get('summary', '')
                review_text = comments + ' ' + summary

                if paper_content.strip() and review_text.strip():
                    data_pairs.append({
                        'input': paper_content.strip(),
                        'output': review_text.strip()
                    })

    return data_pairs


def preprocess_text(text):
    # Remove LaTeX commands
    text = re.sub(r'\$.*?\$', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    text = text.strip()
    return text


def tokenize_and_save(data_pairs, tokenizer, filename):
    tokenized_inputs = []
    tokenized_outputs = []
    for pair in data_pairs:
        # Tokenize input
        inputs = tokenizer(
            pair['input'],
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        # Tokenize output
        outputs = tokenizer(
            pair['output'],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized_inputs.append(inputs)
        tokenized_outputs.append(outputs)
    # Save tokenized data
    torch.save((tokenized_inputs, tokenized_outputs), filename)


def save_data_pairs(data_pairs, filename):
    with open(filename, 'w') as f:
        for pair in data_pairs:
            json_line = json.dumps(pair)
            f.write(json_line + '\n')



if __name__ == '__main__':
    # Define Datasets
    datasets = ['iclr_2017', 'conll_2016', 'arxiv.cs.lg_2007-2017', 'arxiv.cs.cl_2007-2017', 'arxiv.cs.ai_2007-2017', 'acl_2017']

    # Initialize empty lists to collect data pairs from all datasets
    train_data_pairs = []
    dev_data_pairs = []
    test_data_pairs = []

    # Base data directory
    data_dir = 'PeerRead/data/'

    # Loop Over Each Dataset
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        train_path = os.path.join(dataset_dir, 'train')
        dev_path = os.path.join(dataset_dir, 'dev')
        test_path = os.path.join(dataset_dir, 'test')

        print(f"Processing dataset: {dataset}")
        train_pairs = extract_input_output_pairs(train_path)
        dev_pairs = extract_input_output_pairs(dev_path)
        test_pairs = extract_input_output_pairs(test_path)

        print(f"Extracted {len(train_pairs)} training pairs from {dataset}")


        train_data_pairs.extend(train_pairs)
        dev_data_pairs.extend(dev_pairs)
        test_data_pairs.extend(test_pairs)

    print(f"Total training pairs: {len(train_data_pairs)}")
    print(f"Total validation pairs: {len(dev_data_pairs)}")
    print(f"Total test pairs: {len(test_data_pairs)}")

    # Create data directory if it doesn't exist
    preprocessed_data_dir = './processed_data'
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)


    save_data_pairs(train_data_pairs, os.path.join(preprocessed_data_dir, 'train_data.jsonl'))
    save_data_pairs(dev_data_pairs, os.path.join(preprocessed_data_dir, 'dev_data.jsonl'))
    save_data_pairs(test_data_pairs, os.path.join(preprocessed_data_dir, 'test_data.jsonl'))


    print("Preprocessing data...")
    for data_pairs in [train_data_pairs, dev_data_pairs, test_data_pairs]:
        for pair in data_pairs:
            pair['input'] = preprocess_text(pair['input'])
            pair['output'] = preprocess_text(pair['output'])

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    print("Tokenizing and saving data...")
    tokenize_and_save(train_data_pairs, tokenizer, os.path.join(preprocessed_data_dir, 'train_tokenized.pt'))
    tokenize_and_save(dev_data_pairs, tokenizer, os.path.join(preprocessed_data_dir, 'dev_tokenized.pt'))
    tokenize_and_save(test_data_pairs, tokenizer, os.path.join(preprocessed_data_dir, 'test_tokenized.pt'))

    print("Data preparation complete.")
