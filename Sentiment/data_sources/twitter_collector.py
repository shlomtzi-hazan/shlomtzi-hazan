# data_sources/twitter_collector.py
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta

def get_company_tweets(companies, days=7, max_tweets_per_company=100):
    all_tweets = []
    since_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')

    for company in companies:
        query = f"{company} since:{since_date}"
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= max_tweets_per_company:
                break
            all_tweets.append({
                'date': tweet.date,
                'content': tweet.content,
                'company': company
            })

    return pd.DataFrame(all_tweets)
