# social_sentiment_dashboard/app.py
import streamlit as st
from data_sources.twitter_collector import get_company_tweets
from data_sources.news_scraper import get_recent_news
from sentiment_analysis import analyze_sentiment
from utils.visualization import plot_sentiment_over_time, plot_sentiment_comparison
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Set up the app's configuration and title
st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# Load environment variables
load_dotenv()

# Custom CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Orbitron', sans-serif;
            background-color: #1f1f1f;
            color: #a6a4a4;
        }

        .stApp {
            background-color: #0e1f33;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Orbitron', sans-serif;
            color: #00f0ff;
            text-shadow: 0 0 5px #00f0ff, 0 0 10px #00f0ff;
        }

        .block-container {
            padding: 2rem 1rem;
        }

        .stMarkdown, p, span, div {
            color: #e0e0e0 !important;
        }

        .css-1offfwp, .css-1y4p8pa {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        .stPlotlyChart {
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.title("Control Panel")
st.sidebar.subheader("Add Company")
new_company = st.sidebar.text_input("Enter a company name")
if st.sidebar.button("Add"):
    if new_company:
        if "companies" not in st.session_state:
            st.session_state.companies = []
        if new_company not in st.session_state.companies:
            st.session_state.companies.append(new_company)
            st.sidebar.success(f"Added {new_company}")
        else:
            st.sidebar.warning(f"{new_company} already added")

# Time frame filter
st.sidebar.subheader("Select Time Frame")
days_back = st.sidebar.slider("Days to look back", min_value=1, max_value=30, value=7)
start_date = datetime.now() - timedelta(days=days_back)

# Main title
st.title("📊 Brand Sentiment Dashboard")

if "companies" in st.session_state and st.session_state.companies:
    st.markdown(f"### Monitoring: {', '.join(st.session_state.companies)}")

    with st.spinner("Collecting tweets..."):
        tweet_df = get_company_tweets(st.session_state.companies, start_date=start_date)

    tweet_sentiment_df = analyze_sentiment(tweet_df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Sentiment Over Time")
        st.plotly_chart(plot_sentiment_over_time(tweet_sentiment_df), use_container_width=True)

    with col2:
        st.subheader("🏷️ Brand Sentiment Comparison")
        st.plotly_chart(plot_sentiment_comparison(tweet_sentiment_df), use_container_width=True)

    st.markdown("---")
    st.subheader("📰 News Events (Last 24 Hours)")
    news_df = get_recent_news(st.session_state.companies)
    st.dataframe(news_df[['title', 'source', 'published', 'company', 'url']])

else:
    st.info("Add a company in the sidebar to get started.")
