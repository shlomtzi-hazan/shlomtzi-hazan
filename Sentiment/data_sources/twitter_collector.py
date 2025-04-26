# data_sources/twitter_collector.py
import pandas as pd
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import os

from dotenv import load_dotenv
import os
import praw

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD")
)


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🐦 Reddit Collection Logic
def get_company_tweets(companies, days=7, max_posts_per_company=500):
    """
    Fetch posts mentioning a list of companies from Reddit.

    Parameters:
    ----------
    companies : list
        List of company names to fetch posts for.
    days : int
        Number of days to look back for posts.
    max_posts_per_company : int
        Maximum number of posts to fetch per company.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing fetched posts with columns: date, content, company.
    """
    
    all_posts = []
    time_filter = "month"  # Use Reddit's time filters
    # Calculate the exact cutoff date for post-filtering
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    print(f"Searching for posts from the last {days} days (since {cutoff_date})")

    for company in companies:
        try:
            # Search for posts mentioning the company
            subreddit = reddit.subreddit("all")
            query = f"{company}"
            for i, post in enumerate(subreddit.search(query, time_filter=time_filter, limit=max_posts_per_company)):
                if i >= max_posts_per_company:
                    break
                
                # Get post content
                content = post.title + " " + post.selftext
                
                # Simple language check - Skip if empty or contains very few ASCII characters
                if not content or len(content) < 10:
                    continue
                    
                # Check if content is likely English (>60% ASCII characters)
                ascii_chars = sum(1 for c in content if ord(c) < 128)
                if ascii_chars / len(content) < 0.6:
                    continue
                
                all_posts.append({
                    'date': datetime.utcfromtimestamp(post.created_utc),
                    'content': content,
                    'company': company
                })
        except Exception as e:
            print(f"Error fetching posts for {company}: {e}")

    # Convert the list of posts to a DataFrame
    return pd.DataFrame(all_posts)


# 🐦 Original Twitter Collection Logic (disabled for now)
"""
import snscrape.modules.twitter as sntwitter

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
"""

if __name__ == "__main__":
    companies = ["Pepsi", "Coca-Cola", "Olipop"]
    reddit_posts_df = get_company_tweets(companies, days=7, max_posts_per_company=500)
    print(reddit_posts_df.head())
    # reddit_posts_df.to_csv("Downloads/reddit_posts_7days.csv", index=False)
