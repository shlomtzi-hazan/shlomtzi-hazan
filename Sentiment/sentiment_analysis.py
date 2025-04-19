import openai
import pandas as pd

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
    # Set a specific API key for this function
    openai.api_key = "your_specific_api_key_here"  # Replace with the desired API key

    sentiments = []
    for tweet in tweet_df['content']:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Analyze the sentiment of the following tweet:\n\n{tweet}\n\nIs it positive, negative, or neutral?",
                max_tokens=10,
                temperature=0
            )
            sentiment = response['choices'][0]['text'].strip()
            sentiments.append(sentiment)
        except Exception as e:
            print(f"Error analyzing sentiment for tweet: {tweet}\nError: {e}")
            sentiments.append("Error")  # Append "Error" if the API call fails

    tweet_df['sentiment'] = sentiments
    return tweet_df
