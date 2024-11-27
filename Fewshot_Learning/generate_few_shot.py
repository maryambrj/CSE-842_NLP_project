import os
import json
import openai

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


def load_data_pairs(filename):
    data_pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data_pairs.append(json.loads(line))
    return data_pairs


def create_prompt(input_text, examples, max_examples=3):
    prompt = "Evaluate the following paper content and provide a review:\n"
    for example in examples[:max_examples]:
        prompt += f"Paper content:\n{example['input']}\nReview:\n{example['output']}\n\n"
    prompt += f"Paper content:\n{input_text}\nReview:"
    return prompt


def generate_review(input_text, examples):
    prompt = create_prompt(input_text, examples)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",  # or "gpt-4"
                                            messages=[
                                                  {"role": "system",
                                                      "content": "You are a helpful assistant."},
                                                  {"role": "user",
                                                      "content": prompt}
                                            ],
                                            max_tokens=150,
                                            temperature=0.7,
                                            top_p=1.0,
                                            frequency_penalty=0.5,
                                            presence_penalty=0.0)
    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    data_pairs = load_data_pairs('./processed_data/train_data.jsonl')
    results = []

    for i, pair in enumerate(data_pairs[:10]):  # Run a sample (top 10 pairs)
        review = generate_review(pair['input'], data_pairs)
        results.append(
            {'input': pair['input'], 'generated_review': review, 'expected_review': pair['output']})
        print(f"Generated review for sample {i+1}:\n{review}\n")

    # Save generated reviews to a file for analysis
    with open('./processed_data/generated_reviews.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
