import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F
import wandb

class ReviewDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'labels': self.outputs[idx]['input_ids'].squeeze(),
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

if __name__ == '__main__':

    # hyperparameters
    num_epochs = 5
    learning_rate = 5e-5
    batch_size = 2
    
    wandb.init(project='CSE-842_NLP_project', entity='maryam-brj')

    print("Loading tokenized data...")
    train_inputs, train_outputs = torch.load('./processed_data/train_tokenized.pt')
    dev_inputs, dev_outputs = torch.load('./processed_data/dev_tokenized.pt')

    print("Creating datasets and data loaders...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    train_dataset = ReviewDataset(train_inputs, train_outputs)
    dev_dataset = ReviewDataset(dev_inputs, dev_outputs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    print("Initializing the model...")
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

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
            model.save_pretrained('best_model')
            tokenizer.save_pretrained('best_model')
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
    model.save_pretrained('fine_tuned_bart_model')
    tokenizer.save_pretrained('fine_tuned_bart_model')
    print("Model training complete.")

    wandb.finish()
