# social_sentiment_dashboard/app.py
import streamlit as st
from data_sources.twitter_collector import get_company_tweets
from data_sources.news_scraper import get_recent_news
from sentiment_analysis import analyze_sentiment
from utils.visualization import plot_sentiment_over_time, plot_sentiment_comparison
from dotenv import load_dotenv
import os
import openai
from datetime import datetime, timedelta
from utils.post_generator import generate_post

# Set up the app's configuration and title
st.set_page_config(page_title="Brand Sentiment Dashboard", layout="wide")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Apply custom CSS for the new color palette
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Orbitron', sans-serif;
            background-color: #121212; /* Darker true black background */
            color: #e0e0e0; /* Light gray text */
        }

        .stApp {
            background-color: #121212;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Orbitron', sans-serif;
            color: #00f0ff;
            text-shadow: 0 0 5px #00f0ff, 0 0 10px #00f0ff;
        }

        .block-container {
            padding: 2rem 1rem;
        }

        .stMarkdown, .stDataFrame {
            color: #e0e0e0 !important;
        }

        label, .stTextInput label {
            color: #b0b0b0 !important;
        }

        .stTextInput>div>input {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #333;
            border-radius: 5px;
        }

        .stButton>button {
            background-color: #00f0ff;
            color: #121212 !important;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: background-color 0.3s;
        }

        .stButton>button:hover {
            background-color: #00c5d7;
            color: #121212 !important;
        }

        .stAlert-success, .stAlert-success span {
            background-color: rgba(0, 255, 100, 0.15);
            border-left: 5px solid #00ff88;
            color: #dfffe2 !important;
        }

        .stAlert-info, .stAlert-info span {
            background-color: rgba(0, 240, 255, 0.15);
            border-left: 5px solid #00f0ff;
            color: #e0e0e0 !important;
        }

        .css-1offfwp, .css-1y4p8pa, .stDataFrame {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }

        .stPlotlyChart {
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 1rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1c1c1c;
            color: #e0e0e0;
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

    with st.spinner("Collecting posts..."):
        tweet_df = get_company_tweets(st.session_state.companies, days=days_back)

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

    st.markdown("---")
    st.subheader("📝 Generated Posts")

    if not news_df.empty:
        for _, row in news_df.iterrows():
            company = row['company']
            news_title = row['title']
            news_summary = f"{row['source']} published an article titled '{news_title}' on {row['published']}."
            sentiment = tweet_sentiment_df[tweet_sentiment_df['company'] == company]['sentiment'].mean()

            # Determine sentiment label
            if sentiment > 0.05:
                sentiment_label = "positive"
            elif sentiment < -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            # Generate the post
            generated_post = generate_post(company, sentiment_label, news_title, news_summary)

            # Display the generated post
            st.markdown(f"**{company}:**")
            st.write(generated_post)
            st.markdown("---")
    else:
        st.info("No news articles available to generate posts.")

else:
    st.info("Add a company in the sidebar to get started.")
