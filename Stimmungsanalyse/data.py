import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer
import torch


# Load raw data 
def load_data():

    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'raw.csv')
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    dataset_encoding = 'ISO-8859-1'

    df = pd.read_csv(csv_path, encoding= dataset_encoding, names=column_names)

    print("Dataset size:" ,len(df))
    #print(df.head(5))

    return df

# Transform labels to human-readable format
"""def transform_labels(df):
    label_mapping = {0: 'NEGATIVE', 2: 'NEUTRAL', 4: 'POSITIVE'}
    df['target'] = df['target'].map(label_mapping)
    return df"""


#Stopwords not needed for first model

#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()

    #cleaning "@"
    #df['text'] = df['text'].str.replace(r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', case=False, regex=True)

    #keeping "@"
    df['text'] = df['text'].str.replace(r'https?:\S+|http?:\S|[^A-Za-z0-9@]+', ' ', case=False, regex=True)

    df['text'] = df['text'].str.lower()

    # Remove stopwords (optional)
    #df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    print("Cleaned dataset size:", len(df))
    #print(df.head(5))

    return df


df = load_data()
#df_transformed = transform_labels(df)
df_prepared = clean_data(df)

# Save prepared data for futher analysis

output_csv_path = os.path.join(os.path.dirname(__file__), 'data', 'prepared.csv')
df_prepared.to_csv(output_csv_path, index=False)

print(f"Prepared DataFrame saved to {output_csv_path}")









