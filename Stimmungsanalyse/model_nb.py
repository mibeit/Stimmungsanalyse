import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load prepared data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'prepared.csv')
    dataset_encoding = 'ISO-8859-1'
    df = pd.read_csv(csv_path, encoding=dataset_encoding)
    return df

# Preprocess data for new model
def preprocess_data(df):
    # Dropping useless columns
    df = df.drop(columns=['id', 'date', 'flag', 'user'])
    df['target'] = df['target'].replace(4, 1)

    # Remove usernames
    df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)

    # Stopwords need to be cleared
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return df

def tokenize_data(df):
    # Tokenize tweets
    df['tokenized_text'] = df['text'].apply(lambda x: x.split())


    # Feature extraction using CountVectorizer with custom tokenizer
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(df['text'].values.astype('U'))

    return text_counts, cv

def split_data(text_counts, df):
    # Split data into training and testing sets
    X = text_counts
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=19)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Train the model
    model = MultinomialNB(alpha=5.0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    print(accuracy_score(y_test, model.predict(X_test)))
    print(classification_report(y_test, model.predict(X_test)))

def load_predict_data(): 
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'twitter_training.csv')
    column_names = ['tweet_id', 'entity', 'sentiment', 'text']
    dataset_encoding = 'ISO-8859-1'
    df = pd.read_csv(csv_path, encoding=dataset_encoding, names=column_names)
    return df

def clean_predict_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.drop(columns=['entity', 'tweet_id'])
    df['text'] = df['text'].str.replace(r'https?:\S+|http?:\S|[^A-Za-z0-9@]+', ' ', case=False, regex=True)
    df['text'] = df['text'].str.lower()
    df = df[~df['sentiment'].isin(['Irrelevant', 'Neutral'])]

    # Remove usernames
    df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    df['sentiment'] = df['sentiment'].replace({'Positive': 1, 'Negative': 0})

    return df

def predict_new_data(model, cv, df):
    # Preprocess new data
    df = clean_predict_data(df)

    # Tokenize and transform new data
    new_text_counts = cv.transform(df['text'].values.astype('U'))

    # Predict
    predictions = model.predict(new_text_counts)
    df['sentiment_ai'] = predictions
    return df

# Main script
df = load_data()
df_prepared = preprocess_data(df)
text_counts, cv = tokenize_data(df_prepared)
X_train, X_test, y_train, y_test = split_data(text_counts, df_prepared)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

# Load new data for prediction
new_df = load_predict_data()

# Predict new data and add sentiment_ai column
new_df_with_predictions = predict_new_data(model, cv, new_df)

# Count matches and mismatches
matches = (new_df_with_predictions['sentiment'] == new_df_with_predictions['sentiment_ai']).sum()
mismatches = len(new_df_with_predictions) - matches

print(f"Match: {matches}")
print(f"Missmatch: {mismatches}")

# Create pie chart
labels = ['Match', 'Mismatch']
sizes = [matches, mismatches]
colors = ['#66b3ff', '#ff6666']
explode = (0.1, 0)  # Explode the first slice

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of matches and mismatches')
plt.show()