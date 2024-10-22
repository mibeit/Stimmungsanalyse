import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load prepared data 
def load_data():

    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'prepared.csv')
    dataset_encoding = 'ISO-8859-1'

    df = pd.read_csv(csv_path, encoding= dataset_encoding)

    return df

#create graphic for sentiment distribution

def sentiment_distribution(df):

    label_counts = df['target'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title('Verteilung der Stimmungsbewertungen', fontsize=16)
    plt.xlabel('Stimmung', fontsize=12)
    plt.ylabel('Anzahl der Kommentare', fontsize=12)
    plt.show()

#RUN

df = load_data()
sentiment_distribution(df)