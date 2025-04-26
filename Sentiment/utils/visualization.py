# utils/visualization.py
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def plot_sentiment_over_time(df, days_back):
    """
    Plot sentiment over time for multiple companies with debugging information.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing columns: 'date', 'sentiment', 'company'.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        A Plotly figure object for the sentiment over time plot.
    """
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Filter data for the last 'days_back' days
    if days_back:
        cutoff_date = pd.to_datetime('today') - pd.Timedelta(days=days_back)
        df = df[df['date'] >= cutoff_date]

    # Group data by date and company, and calculate the mean sentiment
    df_grouped = df.groupby(['date', 'company'])['sentiment'].mean().reset_index()

    # Plot the data using Plotly
    fig = px.line(df_grouped, x='date', y='sentiment', color='company', markers=True,
                  title='Sentiment Over Time')

    return fig

def plot_sentiment_comparison(df):
    df['date'] = pd.to_datetime(df['date'])
    last_24_hours = datetime.now() - timedelta(hours=24)
    latest = df[df['date'] >= last_24_hours]
    print(latest.head())  # Debugging: print the latest data
    avg_sentiment = latest.groupby('company')['sentiment'].mean().reset_index()
    avg_sentiment['sentiment'] = avg_sentiment['sentiment'].round(3)
    fig = px.bar(avg_sentiment, x='company', y='sentiment', color='company',
                 title='Sentiment Comparison (Last 24h)', text='sentiment')
    return fig
