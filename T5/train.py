import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import BartForConditionalGeneration, BartTokenizer, get_linear_schedule_with_warmup, GenerationConfig
from transformers import T5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup, GenerationConfig
import torch.nn.functional as F
import wandb
import pickle
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence



class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.data[idx]['labels'], dtype=torch.long),
        }

# def collate_fn(batch):
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     attention_mask = torch.stack([item['attention_mask'] for item in batch])
#     labels = torch.stack([item['labels'] for item in batch])

    # return {
    #     'input_ids': input_ids,
    #     'attention_mask': attention_mask,
    #     'labels': labels,
    # }


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Dynamically pad the input_ids, attention_mask, and labels to the length of the longest sequence in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    # For labels, we use `-100` as the padding token, which is the default in PyTorch for ignoring loss computation
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }



if __name__ == '__main__':

    # hyperparameters
    num_epochs = 3
    learning_rate = 5e-6 #1e-5
    batch_size = 2

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"bs{batch_size}_lr{learning_rate}_{current_time}"
    
    wandb.init(project='CSE-842_NLP_project', entity='maryam-brj', name=run_name)


    print("Loading tokenized data...")
    with open('./processed_data/train_tokenized.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./processed_data/dev_tokenized.pkl', 'rb') as f:
        dev_data = pickle.load(f)

    print("Creating datasets and data loaders...")
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    
    train_dataset = ReviewDataset(train_data)
    dev_dataset = ReviewDataset(dev_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    print("Initializing the model...")
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Reset generation config to default to avoid warnings
    # model.generation_config = GenerationConfig.from_pretrained('facebook/bart-base')
    model.generation_config = GenerationConfig()

    print(model.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    wandb.watch(model, log='all')

    # early stopping
    best_dev_loss = float('inf')
    epochs_no_improve = 0
    patience = 2  # Number of epochs to wait before early stopping

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}...")
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            wandb.log({'train_loss': loss.item(), 'epoch': epoch + 1})

            if (step + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation on the dev set
        print("Evaluating on the development set...")
        model.eval()
        total_dev_loss = 0

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_dev_loss += loss.item()

        avg_dev_loss = total_dev_loss / len(dev_loader)
        print(f"Validation loss after epoch {epoch + 1}: {avg_dev_loss:.4f}")

        wandb.log({'val_loss': avg_dev_loss, 'epoch': epoch + 1})

        # Save the best model
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            epochs_no_improve = 0
            model.save_pretrained('./T5/best_model')
            tokenizer.save_pretrained('./T5/best_model')
            print("Best model saved.")
            wandb.run.summary["best_val_loss"] = best_dev_loss
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        # Early stopping check
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        model.train()

    print("Saving the final fine-tuned model...")
    model.save_pretrained('./T5/fine_tuned_model')
    tokenizer.save_pretrained('./T5/fine_tuned_model')
    model.generation_config.save_pretrained('./T5/fine_tuned_model')
    print("Model training complete.")

    wandb.finish()
