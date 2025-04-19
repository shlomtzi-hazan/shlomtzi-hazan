import openai
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def analyze_sentiment(tweet_df):
    """
    Analyze the sentiment of tweets using OpenAI's GPT API.

    Parameters:
    ----------
    tweet_df : pd.DataFrame
        DataFrame containing tweets with a 'content' column.

    Returns:
    -------
    pd.DataFrame
        DataFrame with an additional 'sentiment' column.
    """
    sentiments = []
    for tweet in tweet_df['content']:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Analyze the sentiment of the following tweet:\n\n{tweet}\n\nIs it positive, negative, or neutral?",
            max_tokens=10
        )
        sentiment = response.choices[0].text.strip()
        sentiments.append(sentiment)
    
    tweet_df['sentiment'] = sentiments
    return tweet_df
