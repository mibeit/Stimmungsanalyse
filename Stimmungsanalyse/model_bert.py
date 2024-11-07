import pandas as pd
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import json
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load prepared data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'prepared.csv')
    dataset_encoding = 'ISO-8859-1'
    df = pd.read_csv(csv_path, encoding=dataset_encoding)
    return df

# Tokenize data with fixed padding
def tokenize_and_encode(texts, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,  
            padding='max_length',   
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# Create DataLoader
def create_dataloaders(input_ids, attention_masks, labels, batch_size=32):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    return train_dataloader, val_dataloader

# Initialize model with learning rate scheduler
def initialize_model(total_steps, save_path=None):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    return model, optimizer, scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Train model with early stopping and evaluation metrics
def train_model(model, train_dataloader, optimizer, val_dataloader, scheduler, epochs=3, save_path=None, patience=2, tokenizer=None):
    model.train()
    training_stats = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1}/{epochs}')
        start_time = time.time()
        total_train_loss = 0

        for batch in tqdm(train_dataloader):
            b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

            optimizer.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_stats['train_loss'].append(avg_train_loss)

        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_dataloader, return_stats=True)
        training_stats['val_loss'].append(val_loss)
        training_stats['val_accuracy'].append(val_accuracy)
        training_stats['val_precision'].append(val_precision)
        training_stats['val_recall'].append(val_recall)
        training_stats['val_f1'].append(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
           
            if save_path and not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)  
            logger.info(f'Model & Tokenizer saved to {save_path}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        end_time = time.time()
        logger.info(f'Epoch time: {end_time - start_time:.2f} seconds')

    return training_stats

# Evaluate model with additional metrics
def evaluate_model(model, val_dataloader, return_stats=False):
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    preds_all, labels_all = [], []

    for batch in val_dataloader:
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits
        total_eval_loss += loss.item()

        preds = torch.argmax(F.softmax(logits, dim=1), dim=1).flatten()
        total_eval_accuracy += (preds == b_labels).cpu().numpy().mean()

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(b_labels.cpu().numpy())

    avg_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_loss = total_eval_loss / len(val_dataloader)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='weighted')

    logger.info(f'Accuracy: {avg_accuracy}')
    logger.info(f'Precision: {precision}')
    logger.info(f'Recall: {recall}')
    logger.info(f'F1 Score: {f1}')
    logger.info(f'Loss: {avg_loss}')

    if return_stats:
        return avg_loss, avg_accuracy, precision, recall, f1

# Load model
def load_model(save_path):
    model = BertForSequenceClassification.from_pretrained(save_path)
    tokenizer = BertTokenizer.from_pretrained(save_path)  
    model.to(device)
    return model, tokenizer

# Main function
def main():
    df_prepared = load_data()


    
    label_mapping = {0: 0, 2: 1, 4: 2}
    df_prepared['target'] = df_prepared['target'].map(label_mapping)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks = tokenize_and_encode(df_prepared['text'].values, tokenizer, max_length=128)
    labels = torch.tensor(df_prepared['target'].values, dtype=torch.long)

    train_dataloader, val_dataloader = create_dataloaders(input_ids, attention_masks, labels, batch_size=32)

    total_steps = len(train_dataloader) * 3  # Epochs * train batches
    model, optimizer, scheduler = initialize_model(total_steps)

    # Training with early stopping
    save_path = os.path.join(os.path.dirname(__file__), 'Stimmungsanalyse', 'model_save')
    training_stats = train_model(model, train_dataloader, optimizer, val_dataloader, scheduler, epochs=3, save_path=save_path, tokenizer=tokenizer)

    # load model
    model, tokenizer = load_model(save_path)

    evaluate_model(model, val_dataloader)

    logger.info(f'Training statistics: {training_stats}')

if __name__ == '__main__':
    main()