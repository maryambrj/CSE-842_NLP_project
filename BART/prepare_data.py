import os
import json
import re
import torch
from transformers import BartTokenizer
import pickle

# Define helper functions
def extract_input_output_pairs(data_split_path):
    reviews_path = os.path.join(data_split_path, 'reviews')
    parsed_pdfs_path = os.path.join(data_split_path, 'parsed_pdfs')
    data_pairs = []

    if not os.path.exists(reviews_path) or not os.path.exists(parsed_pdfs_path):
        return data_pairs

    for review_file in os.listdir(reviews_path):
        if review_file.endswith('.json'):
            review_file_path = os.path.join(reviews_path, review_file)
            with open(review_file_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)

            # Extract paper ID from 'id' field in review data
            paper_id = paper_data.get('id', '')

            # Load the parsed PDF content if available
            # The 'name' field in parsed PDFs includes '.pdf', so we need to append '.pdf.json'
            parsed_pdf_file = os.path.join(parsed_pdfs_path, f"{paper_id}.pdf.json")
            if os.path.exists(parsed_pdf_file):
                with open(parsed_pdf_file, 'r', encoding='utf-8') as f_pdf:
                    parsed_pdf = json.load(f_pdf)
                    metadata = parsed_pdf.get('metadata', {})
                    title = metadata.get('title', '')
                    abstract_text = metadata.get('abstractText', '')
                    sections = metadata.get('sections', {})

                    # Check if 'sections' is a dictionary with numbered keys
                    if isinstance(sections, dict):
                        # Iterate over the sections and concatenate headings and texts
                        section_texts = ' '.join([
                            f"{section.get('heading', '')}: {section.get('text', '')}"
                            for key in sorted(sections.keys())
                            for section in [sections[key]]
                            if 'text' in section
                        ])
                    elif isinstance(sections, list):
                        # If 'sections' is a list
                        section_texts = ' '.join([
                            f"{section.get('heading', '')}: {section.get('text', '')}"
                            for section in sections
                            if 'text' in section
                        ])
                    else:
                        section_texts = ''

                    # Combine title, abstract, and sections
                    full_body = f"{title} {abstract_text} {section_texts}".strip()
            else:
                # If parsed PDF file doesn't exist, skip to next review file
                continue

            paper_content = full_body

            # Extract reviews
            reviews = paper_data.get('reviews', {})
            if isinstance(reviews, dict):
                # Iterate over reviews dictionary
                for review_key in sorted(reviews.keys()):
                    review = reviews[review_key]

                    comments = review.get('comments', '').strip()
                    # summary = review.get('summary', '').strip()
                    # review_text = f"{comments} {summary}".strip()

                    review_text = comments

                    if paper_content and review_text:
                        data_pairs.append({
                            'input': paper_content,
                            'output': review_text
                        })
            elif isinstance(reviews, list):
                # If 'reviews' is a list, iterate over it
                for review in reviews:
                    comments = review.get('comments', '').strip()
                    # summary = review.get('summary', '').strip()
                    # review_text = f"{comments} {summary}".strip()

                    review_text = comments

                    if paper_content and review_text:
                        data_pairs.append({
                            'input': paper_content,
                            'output': review_text
                        })

    return data_pairs


def preprocess_text(text):
    # Remove LaTeX commands
    # text = re.sub(r'\$.*?\$', '', text)
    # Remove HTML tags
    # text = re.sub(r'<.*?>', '', text)
    # Remove special characters
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    text = text.strip()
    return text

def tokenize_and_save(data_pairs, tokenizer, filename):
    tokenized_data = []
    for pair in data_pairs:
        # Tokenize input
        inputs = tokenizer(
            pair['input'],
            max_length=1024,  # it is 1024 for BART
            truncation=True,
            padding='max_length',
            # return_tensors='pt'
        )
        # Tokenize output
        outputs = tokenizer(
            pair['output'],
            max_length=1024,
            truncation=True,
            padding='max_length',
            # return_tensors='pt'
        )
        tokenized_data.append({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': outputs['input_ids']
        })

    # Save tokenized data using pickle
    with open(filename, 'wb') as f:
        pickle.dump(tokenized_data, f)


def save_data_pairs(data_pairs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for pair in data_pairs:
            json_line = json.dumps(pair)
            f.write(json_line + '\n')


if __name__ == '__main__':
    # Define Datasets
    datasets = ['iclr_2017', 
                'conll_2016', 
                # 'arxiv.cs.lg_2007-2017', 
                # 'arxiv.cs.cl_2007-2017', 
                # 'arxiv.cs.ai_2007-2017', 
                'acl_2017']

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

    # Create preprocessed_data directory if it doesn't exist
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
    tokenize_and_save(train_data_pairs, tokenizer, os.path.join(preprocessed_data_dir, 'train_tokenized.pkl'))
    tokenize_and_save(dev_data_pairs, tokenizer, os.path.join(preprocessed_data_dir, 'dev_tokenized.pkl'))
    tokenize_and_save(test_data_pairs, tokenizer, os.path.join(preprocessed_data_dir, 'test_tokenized.pkl'))

    print("Data preparation complete.")
