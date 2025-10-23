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

    # We don't know at this point how many topics there are. Therefore, we try training the LDA algorithm at different topic counts, and picking the one that results in the best coherence.
    optimal_coherence = -100
    optimal_lda = None
    optimal_k = 0
    for K in range(10, 50):

        # Train LDA model. We want to determine how we can best split the data into 4 topics
        lda = LdaModel(corpus, num_topics=K, id2word=dictionary,
                       passes=10, random_state=2)

        # Now that the LDA model is done, let's see how good it is by computing its 'coherence score'
        coherence_model = CoherenceModel(
            model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        if (coherence_score > optimal_coherence):
            print(
                f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is the best so far!')
            optimal_coherence = coherence_score
            optimal_lda = lda
            optimal_k = K
        else:
            print(
                f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is not very good.')

    # Okay, we tried many topic numbers and selected the best one. Let's see how our trained LDA model for the optimal number of topics performed.

    # First, to see the topics, print top 5 most representative words per topic
    print(
        f'These are the words most representative of each of the {optimal_k} topics:')
    for i, topic in optimal_lda.print_topics(num_words=5):
        print(f"Topic {i}: {topic}")

    # Then, let's determine how many posts we have for each topic
    # Count the dominant topic for each document
    topic_counts = [0] * optimal_k  # one counter per topic
    for bow in corpus:
        topic_dist = optimal_lda.get_document_topics(
            bow)  # list of (topic_id, probability)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[
            0]  # find the top probability
        # add 1 to the most probable topic's counter
        topic_counts[dominant_topic] += 1

    # Display the topic counts
    for i, count in enumerate(topic_counts):
        print(f"Topic {i}: {count} posts")


if __name__ == '__main__':
    with app.app_context():
        main()

# printtaa 10 eniten postauksia sisältävää topicia + määrät vikassa display funktiossa
