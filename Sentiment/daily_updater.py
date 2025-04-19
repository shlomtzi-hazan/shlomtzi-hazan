# daily_updater.py
from data_sources.twitter_collector import get_company_tweets
from data_sources.news_scraper import get_recent_news
from sentiment_analysis import analyze_sentiment
import pandas as pd
import os

def update_data():
    companies = ["Pepsi", "Coca-Cola", "Olipop"]
    
    # Update tweets
    tweets_df = get_company_tweets(companies)
    tweets_df = analyze_sentiment(tweets_df)
    tweets_df.to_csv("data/tweets.csv", index=False)

    # Update news
    news_df = get_recent_news(companies)
    news_df.to_csv("data/news.csv", index=False)

if __name__ == "__main__":
    update_data()
