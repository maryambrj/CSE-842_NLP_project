import os
import json
import re

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

            paper_id = paper_data.get('id', '')
            parsed_pdf_file = os.path.join(parsed_pdfs_path, f"{paper_id}.pdf.json")
            if os.path.exists(parsed_pdf_file):
                with open(parsed_pdf_file, 'r', encoding='utf-8') as f_pdf:
                    parsed_pdf = json.load(f_pdf)
                    metadata = parsed_pdf.get('metadata', {})
                    title = metadata.get('title', '')
                    abstract_text = metadata.get('abstractText', '')
                    sections = metadata.get('sections', [])

                    section_texts = ' '.join([
                        f"{section.get('heading', '')}: {section.get('text', '')}"
                        for section in sections if 'text' in section
                    ])
                    full_body = f"{title} {abstract_text} {section_texts}".strip()
            else:
                continue

            paper_content = full_body
            reviews = paper_data.get('reviews', [])

            for review in reviews:
                review_text = review.get('comments', '').strip()
                if paper_content and review_text:
                    data_pairs.append({
                        'input': paper_content,
                        'output': review_text
                    })

    return data_pairs

def save_data_pairs(data_pairs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for pair in data_pairs:
            json_line = json.dumps(pair)
            f.write(json_line + '\n')

if __name__ == '__main__':
    datasets = ['iclr_2017', 'conll_2016', 'acl_2017']
    data_dir = 'PeerRead/data/'
    all_data_pairs = []

    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        train_path = os.path.join(dataset_dir, 'train')
        train_pairs = extract_input_output_pairs(train_path)
        all_data_pairs.extend(train_pairs)

    save_data_pairs(all_data_pairs, './processed_data/train_data.jsonl')
    print("Data extraction complete.")
