import tweepy
import pandas as pd
import praw
from dotenv import load_dotenv
import time
import os
import re
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib  # Import joblib for saving the model
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn import datasets


# Twitter API credentials
load_dotenv()

consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("Tweet_bearer")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Reddit API credentials
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


# Data Preprocessing
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()


# Initialize Tweepy client for v2
client = tweepy.Client(bearer_token)


def fetch_tweets(query, count=10):
    # Use X API v2 for fetching recent tweets
    response = client.search_recent_tweets(query=query, max_results=count, tweet_fields=['created_at'])
    tweet_data = []
    
    if response.data:
        for tweet in response.data:
            tweet_data.append({'text': tweet.text, 'created_at': tweet.created_at})
    else:
        print("No tweets found.")
    
    # Convert the fetched data into a DataFrame
    df = pd.DataFrame(tweet_data)
    df['cleaned_text'] = df['text'].apply(clean_text)  # Apply your text cleaning function
    return df

# Function to fetch Reddit posts
def fetch_reddit_posts(subreddit_name, limit=5):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.hot(limit=limit):
        posts.append({'title': submission.title, 'score': submission.score, 'url': submission.url, 'text': submission.selftext})
    return posts


# Sentiment Analysis Model
vectorizer = TfidfVectorizer()
model = LogisticRegression(random_state=0)


# Train Model Function
def train_model():
    data = pd.read_csv('D:/Vscode_projs/KYNproj/Reddit_Data.csv')
    x1 = data['clean_comment'].fillna('').astype(str)
    y = data['category']
    x1 = x1[:len(y)]

    vectorizer = CountVectorizer()

    # Fit and transform the data
    xcv = vectorizer.fit_transform(x1)
    x_train, x_test, y_train, y_test = train_test_split(xcv, y, test_size=0.3, random_state=23)
    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)

    # Prediction and Metrics
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the model and vectorizer
    joblib.dump(clf, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')


# Function for sentiment analysis using the saved model
def analyze_sentiment(texts):
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    vectors = vectorizer.transform(texts)
    predictions = model.predict(vectors)
    return predictions

#