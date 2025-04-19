# social_sentiment_dashboard/app.py
import streamlit as st
from data_sources.twitter_collector import get_company_tweets
from data_sources.news_scraper import get_recent_news
from sentiment_analysis import analyze_sentiment
from utils.visualization import plot_sentiment_over_time, plot_sentiment_comparison
from dotenv import load_dotenv
import os

# Set up the app's configuration and title (must be the first Streamlit command)
st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# Load environment variables
load_dotenv()

# Apply custom CSS for a professional look
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
            color: #00f0ff;  /* Neon cyan */
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


# Title of the dashboard
st.title("Brand Sentiment Dashboard")

# Initialize session state for companies
if "companies" not in st.session_state:
    st.session_state.companies = []

# Input field to add a company
st.subheader("Add a Company")
new_company = st.text_input("Enter a company name to analyze:")
if st.button("Add Company"):
    if new_company and new_company not in st.session_state.companies:
        st.session_state.companies.append(new_company)
        st.success(f"Added {new_company} to the list!")
    elif new_company in st.session_state.companies:
        st.warning(f"{new_company} is already in the list!")
    else:
        st.error("Please enter a valid company name.")

# Display the current list of companies
st.subheader("Companies to Monitor")
if st.session_state.companies:
    st.write(", ".join(st.session_state.companies))
else:
    st.write("No companies added yet. Please add a company to start.")

# Scrape tweets if there are companies in the list
if st.session_state.companies:
    st.subheader("Tweet Sentiment Analysis")
    st.write("Analyzing tweets for the following companies: ", ", ".join(st.session_state.companies))
    tweet_df = get_company_tweets(st.session_state.companies)

    # Analyze sentiment
    tweet_sentiment_df = analyze_sentiment(tweet_df)

    # Plot 1: Sentiment Over Time
    st.subheader("Sentiment Over Time")
    st.plotly_chart(plot_sentiment_over_time(tweet_sentiment_df), use_container_width=True)

    # Plot 2: Comparison Bar Chart
    st.subheader("Brand Sentiment Comparison")
    st.plotly_chart(plot_sentiment_comparison(tweet_sentiment_df), use_container_width=True)

    # News Event Impact
    st.subheader("News Events (Last 24 Hours)")
    news_df = get_recent_news(st.session_state.companies)
    st.dataframe(news_df[['title', 'source', 'published', 'company', 'url']])
else:
    st.info("Please add at least one company to analyze.")
