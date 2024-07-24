import os

# Download the dataset using Kaggle API
os.system('kaggle datasets download -d tmdb/tmdb-movie-metadata')

# Unzip the dataset
os.system('unzip tmdb-movie-metadata.zip')

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
movies_metadata = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on movie ID
movies = movies_metadata.merge(credits, left_on='id', right_on='movie_id', suffixes=('', '_credits'))

# Extract relevant columns
data = movies[['overview', 'genres']]

# Preprocessing: Convert genres to a single label (assuming first genre)
data['genres'] = data['genres'].apply(lambda x: ast.literal_eval(x)[0]['name'] if len(ast.literal_eval(x)) > 0 else 'Unknown')

# Remove rows with unknown genres
data = data[data['genres'] != 'Unknown']
# Preprocessing: Convert genres to a single label (assuming first genre)
import json  # Import json module for handling JSON-like strings

def extract_first_genre(genre_str):
    try:
        genres = json.loads(genre_str.replace("'", "\""))  # Convert string to list of dictionaries
        return genres[0]['name'] if genres else 'Unknown'
    except:
        return 'Unknown'

data['genres'] = data['genres'].apply(extract_first_genre)

# Text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Handle potential float values or missing data
    if isinstance(text, str):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    else:
        return ''  # Return an empty string for non-string values

data['cleaned_overview'] = data['overview'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_overview'], data['genres'], test_size=0.2, random_state=42)

# Build a text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB()),
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
