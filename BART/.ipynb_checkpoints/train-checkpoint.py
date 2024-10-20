import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer

class ReviewDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.input_ids = torch.cat([item['input_ids'] for item in inputs], dim=0)
        self.attention_mask = torch.cat([item['attention_mask'] for item in inputs], dim=0)
        self.labels = torch.cat([item['input_ids'] for item in outputs], dim=0)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }

if __name__ == '__main__':
    
    print("Loading tokenized data...")
    train_inputs, train_outputs = torch.load('train_tokenized.pt')
    dev_inputs, dev_outputs = torch.load('dev_tokenized.pt')

    print("Creating datasets and data loaders...")
    train_dataset = ReviewDataset(train_inputs, train_outputs)
    dev_dataset = ReviewDataset(dev_inputs, dev_outputs)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=2)

    print("Initializing the model...")
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed.")

        # Add evaluation on the dev set here

    print("Saving the fine-tuned model...")
    model.save_pretrained('fine_tuned_bart_model')
    print("Model training complete.")
