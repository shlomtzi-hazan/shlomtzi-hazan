# utils/visualization.py
import pandas as pd
import plotly.express as px

def plot_sentiment_over_time(df):
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

    # Group data by date and company, and calculate the mean sentiment
    df_grouped = df.groupby(['date', 'company'])['sentiment'].mean().reset_index()

    # Plot the data using Plotly
    fig = px.line(df_grouped, x='date', y='sentiment', color='company', markers=True,
                  title='Sentiment Over Time')

    return fig

def plot_sentiment_comparison(df):
    latest = df[df['date'] == df['date'].max()]
    avg_sentiment = latest.groupby('company')['sentiment'].mean().reset_index()
    fig = px.bar(avg_sentiment, x='company', y='sentiment', color='company',
                 title='Sentiment Comparison (Last 24h)', text='sentiment')
    return fig
