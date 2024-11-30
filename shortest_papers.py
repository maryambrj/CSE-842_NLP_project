import json

file_path = "./processed_data/test_data.jsonl"

unique_data = {}
with open(file_path, 'r') as file:
    for idx, line in enumerate(file):
        data_point = json.loads(line.strip())
        input_text = data_point.get("input", "")  
        if input_text not in unique_data:
            unique_data[input_text] = (idx, data_point) 

unique_data_list = [(index, len(data["input"].split()), data) for index, data in unique_data.values()]

shortest_inputs = sorted(unique_data_list, key=lambda x: x[1])[:10]

print("Indices and Lengths of 10 Papers with the Shortest Unique 'Input':")
for rank, (original_index, length, entry) in enumerate(shortest_inputs, 1):
    print(f"Rank {rank}: Original Index: {original_index}, Length (words): {length}")