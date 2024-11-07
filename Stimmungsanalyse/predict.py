import os
import sys
import torch
import pandas as pd
import time
from transformers import BertTokenizer, BertForSequenceClassification


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Stimmungsanalyse.model_bert import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'twitter_training.csv')
    column_names = ['tweet_id', 'entity', 'sentiment', 'text']
    dataset_encoding = 'ISO-8859-1'
    df = pd.read_csv(csv_path, encoding=dataset_encoding, names=column_names)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.drop(columns=['entity', 'tweet_id'])
    df['text'] = df['text'].str.replace(r'https?:\S+|http?:\S|[^A-Za-z0-9@]+', ' ', case=False, regex=True)
    df['text'] = df['text'].str.lower()
    df = df[~df['sentiment'].isin(['Irrelevant', 'Neutral'])]
    return df

label_dict = {
    0: "Negative",
    1: "neutral",
    2: "Positive"
}

def predict(text, model, tokenizer):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    predicted_label_meaning = label_dict[predicted_label]
    return predicted_label_meaning

def classify_texts(df, model_path='Stimmungsanalyse/data/model_save'):
    model, tokenizer = load_model(model_path)
    df['sentiment_ai'] = df['text'].apply(lambda x: predict(x, model, tokenizer))
    return df

if __name__ == "__main__":
    df = load_data()
    clean_df = clean_data(df)
    classified_df = classify_texts(clean_df)

    matches = (classified_df['sentiment'] == classified_df['sentiment_ai']).sum()
    mismatches = len(classified_df) - matches

    print(f"Übereinstimmungen: {matches}")
    print(f"Nicht-Übereinstimmungen: {mismatches}")