# utils/visualization.py
import plotly.express as px
import pandas as pd

def plot_sentiment_over_time(df):
    df_grouped = df.groupby(['date', 'company'])['sentiment'].mean().reset_index()
    fig = px.line(df_grouped, x='date', y='sentiment', color='company', markers=True,
                  title='Sentiment Over Time')
    return fig

def plot_sentiment_comparison(df):
    latest = df[df['date'] == df['date'].max()]
    avg_sentiment = latest.groupby('company')['sentiment'].mean().reset_index()
    fig = px.bar(avg_sentiment, x='company', y='sentiment', color='company',
                 title='Sentiment Comparison (Last 24h)', text='sentiment')
    return fig
