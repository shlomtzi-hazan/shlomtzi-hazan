from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(df):
    df['sentiment'] = df['content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
