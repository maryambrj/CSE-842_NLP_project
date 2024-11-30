import torch
from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig
from torch.nn.utils.rnn import pad_sequence

def collate_fn_manual(input_text):

    input_ids = tokenizer.encode(input_text, truncation=True, max_length=512, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask

if __name__ == '__main__':
    model_dir = './BART/best_model'
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()


    generation_config = GenerationConfig(
        early_stopping=True,
        num_beams=4,
        no_repeat_ngram_size=5, #3,
        max_length=512,  
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    input_text = "Sample essay text."
    
    input_ids, attention_mask = collate_fn_manual(input_text)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=generation_config.num_beams,
            no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
            max_length=generation_config.max_length,
            early_stopping=generation_config.early_stopping,
            decoder_start_token_id=generation_config.decoder_start_token_id,
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Generated Output:")
    print(decoded_output)
