import os
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def extract_data_for_analysis(data_split_path):
    reviews_path = os.path.join(data_split_path, 'reviews')
    parsed_pdfs_path = os.path.join(data_split_path, 'parsed_pdfs')
    data_list = []

    if not os.path.exists(reviews_path) or not os.path.exists(parsed_pdfs_path):
        return data_list

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

                    # Extract sections text
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

                    # Combine title, abstract, and sections
                    full_body = f"{title} {abstract_text} {section_texts}".strip()
            else:
                # If parsed PDF file doesn't exist, skip to next review file
                continue

            paper_content = full_body

            # Extract acceptance status
            accepted = paper_data.get('accepted', False)

            # Extract reviews
            reviews = paper_data.get('reviews', {})
            review_texts = []
            if isinstance(reviews, dict):
                for review_key in sorted(reviews.keys()):
                    review = reviews[review_key]
                    comments = review.get('comments', '').strip()
                    if comments:
                        review_texts.append(comments)
            elif isinstance(reviews, list):
                for review in reviews:
                    comments = review.get('comments', '').strip()
                    if comments:
                        review_texts.append(comments)

            data_list.append({
                'paper_id': paper_id,
                'accepted': accepted,
                'paper_content': paper_content,
                'reviews': review_texts
            })

    return data_list

def count_latex_and_html(text):
    latex_commands = re.findall(r'\$.*?\$', text)
    html_tags = re.findall(r'<.*?>', text)
    return len(latex_commands), len(html_tags)

def count_special_characters(text):
    # Define a pattern for special characters (excluding common punctuation)
    special_chars = re.findall(r'[^a-zA-Z0-9\s\.\,\;\:\!\?\-\'\"\/\(\)\[\]\{\}]', text)
    return len(special_chars)

def count_math_symbols(text):
    # Unicode range for mathematical operators and symbols
    math_symbols = re.findall(r'[\u2200-\u22FF]', text)
    return len(math_symbols)

def analyze_dataset(datasets, data_dir):
    total_papers = 0
    accepted_papers = 0
    rejected_papers = 0
    paper_lengths = []
    review_lengths = []
    vocabulary = Counter()

    total_latex_commands = 0
    total_html_tags = 0
    total_special_chars = 0
    total_math_symbols = 0

    for dataset in datasets:
        print(f"Analyzing dataset: {dataset}")
        for split in ['train', 'dev', 'test']:
            data_split_path = os.path.join(data_dir, dataset, split)
            data_list = extract_data_for_analysis(data_split_path)
            for data in data_list:
                total_papers += 1
                if data['accepted']:
                    accepted_papers += 1
                else:
                    rejected_papers += 1

                # Paper length (in words)
                paper_length = len(data['paper_content'].split())
                paper_lengths.append(paper_length)

                # Update vocabulary and count special characters
                words = data['paper_content'].split()
                vocabulary.update(words)

                latex_cmds, html_tags = count_latex_and_html(data['paper_content'])
                total_latex_commands += latex_cmds
                total_html_tags += html_tags

                special_chars = count_special_characters(data['paper_content'])
                total_special_chars += special_chars

                math_symbols = count_math_symbols(data['paper_content'])
                total_math_symbols += math_symbols

                for review in data['reviews']:
                    # Review length (in words)
                    review_length = len(review.split())
                    review_lengths.append(review_length)

                    # Update vocabulary and count special characters
                    words = review.split()
                    vocabulary.update(words)

                    latex_cmds, html_tags = count_latex_and_html(review)
                    total_latex_commands += latex_cmds
                    total_html_tags += html_tags

                    special_chars = count_special_characters(review)
                    total_special_chars += special_chars

                    math_symbols = count_math_symbols(review)
                    total_math_symbols += math_symbols

    # Frequency Assessment
    print(f"Total LaTeX commands found: {total_latex_commands}")
    print(f"Total HTML tags found: {total_html_tags}")
    print(f"Total special characters found: {total_special_chars}")
    print(f"Total math symbols found: {total_math_symbols}")

    # Class Imbalance
    print(f"Total papers: {total_papers}")
    print(f"Accepted papers: {accepted_papers}")
    print(f"Rejected papers: {rejected_papers}")

    # Plotting histograms
    plot_histogram(paper_lengths, 'Paper Lengths (in words)', 'Number of Papers', 'Distribution of Paper Lengths', 'paper_lengths_histogram.png')
    plot_histogram(review_lengths, 'Review Lengths (in words)', 'Number of Reviews', 'Distribution of Review Lengths', 'review_lengths_histogram.png')

    # Vocabulary Analysis
    most_common_words = vocabulary.most_common(50)
    print("\nMost common words in the dataset:")
    for word, freq in most_common_words:
        print(f"{word}: {freq}")

def plot_histogram(data, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10,6))
    plt.hist(data, bins=50, color='blue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Histogram saved as {filename}")

if __name__ == '__main__':

    datasets = [
        'iclr_2017', 
        'conll_2016', 
        # 'arxiv.cs.lg_2007-2017', 
        # 'arxiv.cs.cl_2007-2017', 
        # 'arxiv.cs.ai_2007-2017', 
        'acl_2017'
    ]

    data_dir = 'PeerRead/data/'

    analyze_dataset(datasets, data_dir)
