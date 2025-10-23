from flask import Flask, render_template, request, redirect, url_for, session, flash, g
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet
import collections
import json
import sqlite3
import hashlib
import re
from datetime import datetime
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = '123456789'
DATABASE = 'database.sqlite'

def get_db():
    """
    Connect to the application's configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            DATABASE,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('vader_lexicon')

    db = get_db()
    cursor = db.cursor()
    data_frame = pd.read_sql_query('SELECT content FROM posts', db)
    data = pd.DataFrame(data_frame)

    stop_words = stopwords.words('english')
    stop_words.extend(['day', 'one', 'today', 'finally', 'like', 'see', 'incredible', 'would', 'best', 'always', 'amazing', 'bought', 'quick' 'people', 'new', 'fun', 'think', 'know', 'believe', 'many', 'thing', 'need', 'small', 'even', 'make', 'love',
                        'mean', 'fact', 'question', 'time', 'reason', 'also', 'could', 'true', 'well',  'life', 'said', 'year', 'going', 'good', 'really', 'much', 'want', 'back', 'look', 'article', 'host', 'university', 'reply', 'thanks', 'mail', 'post', 'please'])

    lemmatizer = WordNetLemmatizer()

    bow_list = []

    for _, row in data.iterrows():
        text = row['content']
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 2]
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        if len(tokens) > 0:
            bow_list.append(tokens)

    dictionary = Dictionary(bow_list)
    dictionary.filter_extremes(no_below=2, no_above=0.25)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]

    K = 10

    lda = LdaModel(corpus, num_topics=K, id2word=dictionary,
                    passes=10, random_state=2)

    coherence_model = CoherenceModel(
        model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    print(
        f'These are the words most representative of each of the {K} topics:')
    for i, topic in lda.print_topics(num_words=1):
        print(f"Topic {i}: {topic}")

    topic_counts = [0] * K
    for bow in corpus:
        topic_dist = lda.get_document_topics(bow)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        topic_counts[dominant_topic] += 1

    for i, count in enumerate(topic_counts):
        print(f"Topic {i}: {count} posts")

if __name__ == '__main__':
    main()