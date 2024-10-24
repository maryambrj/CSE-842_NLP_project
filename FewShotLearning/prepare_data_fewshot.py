import os
import json
import re
import random

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
            parsed_pdf_file = os.path.join(parsed_pdfs_path, f"{paper_id}.pdf.json")
            if os.path.exists(parsed_pdf_file):
                with open(parsed_pdf_file, 'r', encoding='utf-8') as f_pdf:
                    parsed_pdf = json.load(f_pdf)
                    metadata = parsed_pdf.get('metadata', {})
                    title = metadata.get('title', '')
                    abstract_text = metadata.get('abstractText', '')
                    sections = metadata.get('sections', {})

                    # Process sections
                    if isinstance(sections, dict):
                        section_texts = ' '.join([
                            f"{section.get('heading', '')}: {section.get('text', '')}"
                            for key in sorted(sections.keys())
                            for section in [sections[key]]
                            if 'text' in section
                        ])
                    elif isinstance(sections, list):
                        section_texts = ' '.join([
                            f"{section.get('heading', '')}: {section.get('text', '')}"
                            for section in sections
                            if 'text' in section
                        ])
                    else:
                        section_texts = ''

                    full_body = f"{title} {abstract_text} {section_texts}".strip()
            else:
                continue  

            paper_content = full_body

            # Extract reviews
            reviews = paper_data.get('reviews', {})
            if isinstance(reviews, dict):
                # Iterate over reviews dictionary
                for review_key in sorted(reviews.keys()):
                    review = reviews[review_key]

                    comments = review.get('comments', '').strip()
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
                    review_text = comments

                    if paper_content and review_text:
                        data_pairs.append({
                            'input': paper_content,
                            'output': review_text
                        })

    return data_pairs

def preprocess_text(text):
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def save_data_pairs(data_pairs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for pair in data_pairs:
            json_line = json.dumps(pair)
            f.write(json_line + '\n')

if __name__ == '__main__':
    # Define Datasets
    datasets = ['iclr_2017', 'conll_2016', 'acl_2017']

    # Initialize empty lists to collect data pairs from all datasets
    train_data_pairs = []
    dev_data_pairs = []
    test_data_pairs = []

    # Base data directory
    data_dir = 'PeerRead/data/'

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

    print("Preprocessing data...")
    for data_pairs in [train_data_pairs, dev_data_pairs, test_data_pairs]:
        for pair in data_pairs:
            pair['input'] = preprocess_text(pair['input'])
            pair['output'] = preprocess_text(pair['output'])

    processed_data_dir = './processed_data'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)


    save_data_pairs(train_data_pairs, os.path.join(processed_data_dir, 'train_data.jsonl'))
    save_data_pairs(dev_data_pairs, os.path.join(processed_data_dir, 'dev_data.jsonl'))
    save_data_pairs(test_data_pairs, os.path.join(processed_data_dir, 'test_data.jsonl'))


    few_shot_examples = random.sample(train_data_pairs, k=5)  # Choose 5 examples for few-shot learning

    # few-shot examples- saved ones
    save_data_pairs(few_shot_examples, os.path.join(processed_data_dir, 'few_shot_examples.jsonl'))

    print("Data preparation for few-shot learning complete.")
