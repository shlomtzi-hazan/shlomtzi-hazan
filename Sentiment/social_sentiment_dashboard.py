# social_sentiment_dashboard/app.py
import streamlit as st
from data_sources.twitter_collector import get_company_tweets
from data_sources.news_scraper import get_recent_news
from sentiment_analysis import analyze_sentiment
from utils.visualization import plot_sentiment_over_time, plot_sentiment_comparison
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Brand Buzz Dashboard", layout="wide")
st.title("📊 Brand Sentiment Dashboard")

# Companies to monitor
companies = ["Pepsi", "Coca-Cola", "Olipop"]

# Scrape tweets
tweet_df = get_company_tweets(companies)

# Analyze sentiment
tweet_sentiment_df = analyze_sentiment(tweet_df)

# Plot 1: Sentiment Over Time
st.subheader("📈 Sentiment Over Time")
st.plotly_chart(plot_sentiment_over_time(tweet_sentiment_df), use_container_width=True)

# Plot 2: Comparison Bar Chart
st.subheader("📊 Brand Sentiment Comparison")
st.plotly_chart(plot_sentiment_comparison(tweet_sentiment_df), use_container_width=True)

# News Event Impact
st.subheader("📰 News Events (last 24h)")
news_df = get_recent_news(companies)
st.dataframe(news_df[['title', 'source', 'published', 'company', 'url']])
