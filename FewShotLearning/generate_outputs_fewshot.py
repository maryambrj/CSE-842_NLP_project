import os
import json
import re
import openai
import tiktoken  

openai.api_key = os.getenv('OPENAI_API_KEY') # set from .dotenv file
encoding = tiktoken.encoding_for_model('text-davinci-003')


def preprocess_text(text):
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_data_pairs(filename):
    data_pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            pair = json.loads(line.strip())
            data_pairs.append(pair)
    return data_pairs

def construct_prompt(input_text, few_shot_examples, max_context_length=4096, max_response_tokens=150):
    few_shot_prompt = ""
    for example in few_shot_examples:
        few_shot_prompt += f"Text: {example['input']}\nSummary: {example['output']}\n\n"
    
    prompt = few_shot_prompt + f"Text: {input_text}\nSummary:"
    
    prompt_tokens = encoding.encode(prompt)
    
    if len(prompt_tokens) + max_response_tokens > max_context_length:
        # Calculate available tokens for the input_text
        few_shot_tokens = encoding.encode(few_shot_prompt + "Text: \nSummary:")
        available_tokens_for_input = max_context_length - len(few_shot_tokens) - max_response_tokens
        
        # Truncate the input_text to fit into available tokens
        input_tokens = encoding.encode(input_text)
        if len(input_tokens) > available_tokens_for_input:
            input_tokens = input_tokens[:available_tokens_for_input]
            input_text = encoding.decode(input_tokens)
            prompt = few_shot_prompt + f"Text: {input_text}\nSummary:"
    return prompt

def generate_outputs(data_pairs, few_shot_examples, output_file):
    all_outputs = []
    all_references = []
    
    for idx, pair in enumerate(data_pairs):
        input_text = pair['input']
        reference_summary = pair['output']

        input_text = preprocess_text(input_text)

        prompt = construct_prompt(input_text, few_shot_examples)

        try:
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                max_tokens=150,
                temperature=0.7,
                n=1,
                stop=None,
            )
            generated_summary = response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating output for example {idx}: {e}")
            generated_summary = ""

        all_outputs.append(generated_summary)
        all_references.append(reference_summary)

        # Optionally print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(data_pairs)} examples.")

    # Save the generated summaries
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in all_outputs:
            f.write(output + '\n')

    ref_file = output_file.replace('.txt', '_references.txt')
    with open(ref_file, 'w', encoding='utf-8') as f:
        for ref in all_references:
            f.write(ref + '\n')

if __name__ == '__main__':
    # Check for OpenAI API key
    if openai.api_key is None:
        print("Please set your OpenAI API key")
        exit(1)

    few_shot_examples_file = './processed_data/few_shot_examples.jsonl'
    if not os.path.exists(few_shot_examples_file):
        print("Few-shot examples file not found")
        exit(1)
    few_shot_examples = load_data_pairs(few_shot_examples_file)

    # Load dev and test data
    dev_data_file = './processed_data/dev_data.jsonl'
    test_data_file = './processed_data/test_data.jsonl'

    if not os.path.exists(dev_data_file) or not os.path.exists(test_data_file):
        print("Dev or test data files not found. Please prepare the data.")
        exit(1)

    dev_data = load_data_pairs(dev_data_file)
    test_data = load_data_pairs(test_data_file)

    print("Generating outputs for the dev set...")
    output_file = './GPT3/dev_outputs.txt'
    generate_outputs(dev_data, few_shot_examples, output_file)
    print(f"Dev outputs saved to {output_file}")

    print("Generating outputs for the test set...")
    output_file = './GPT3/test_outputs.txt'
    generate_outputs(test_data, few_shot_examples, output_file)
    print(f"Test outputs saved to {output_file}")

    print("Generation complete.")
