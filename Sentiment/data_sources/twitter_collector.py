# data_sources/twitter_collector.py
import pandas as pd
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt_tweets(company, n=100):
    """
    Generate realistic tweets mentioning a company using OpenAI GPT.
    Falls back to mock tweets in case of an error.

    Parameters:
    ----------
    company : str
        The name of the company to generate tweets for.
    n : int
        The number of tweets to generate.

    Returns:
    -------
    list
        A list of generated tweets as strings.
    """
    try:
        prompt = (
            f"Generate {n} realistic tweets mentioning the company '{company}'. "
            f"Make them sound like they were posted by real users on social media, you can look online for insperation. "
            f"Each tweet should be on a new line."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        # Extract the content from the response
        tweets = response['choices'][0]['message']['content'].split('\n')
        tweets = [t.strip('- ').strip() for t in tweets if t.strip()]
        return tweets[:n]
    except Exception as e:
        print(f"OpenAI error for {company}: {e}")
        return [f"Mock tweet for {company} #{i+1}" for i in range(n)]

def get_company_tweets(companies, days=7, max_tweets_per_company=500):
    """
    Generate tweets for a list of companies using GPT or mock data.

    Parameters:
    ----------
    companies : list
        List of company names to generate tweets for.
    days : int
        Number of days to look back for tweets (not used in GPT generation).
    max_tweets_per_company : int
        Maximum number of tweets to generate per company.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing generated tweets with columns: date, content, company.
    """
    all_tweets = []
    for company in companies:
        # Generate tweets for the company
        tweets = generate_gpt_tweets(company, max_tweets_per_company)
        for i, content in enumerate(tweets):
            # Calculate hours back for this tweet based on days parameter
            total_hours = days * 24
            hours_per_tweet = max(1, total_hours // max_tweets_per_company)
            
            # Distribute tweets over the entire time period, at least one per hour
            tweet_hours_ago = i * hours_per_tweet
            
            all_tweets.append({
                'date': datetime.utcnow() - timedelta(hours=tweet_hours_ago),
                'content': content,
                'company': company
            })
    # Convert the list of tweets to a DataFrame
    return pd.DataFrame(all_tweets)


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
