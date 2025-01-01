# app.py
from flask import Flask, request, jsonify, render_template
from data_processing import fetch_tweets, fetch_reddit_posts, clean_text, analyze_sentiment

# Flask App Setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form.get('query')

    # Collect data from multiple sources
    tweet_df = fetch_tweets(query, count=10)
    reddit_posts = fetch_reddit_posts('all', limit=5)

    # Prepare combined dataset
    all_texts = list(tweet_df['cleaned_text'])
    all_texts.extend([clean_text(post['text']) for post in reddit_posts])

    # Perform sentiment analysis
    sentiments = analyze_sentiment(all_texts)
    tweet_df['sentiment'] = sentiments[:len(tweet_df)]
    tweet_df['sentiment_label'] = tweet_df['sentiment'].apply(lambda x: 'Positive' if x == 1 else 'Negative')

    sentiment_counts = tweet_df['sentiment_label'].value_counts()
    insights = "Positive sentiment is higher." if sentiment_counts.get('Positive', 0) > sentiment_counts.get('Negative', 0) else "Negative sentiment is higher."

    return jsonify({'insights': insights, 'sentiment_counts': sentiment_counts.to_dict()})

if __name__ == '__main__':
    app.run(debug=True)
