import openai
from dotenv import load_dotenv
import os

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_post(company, sentiment, news_title, news_summary):
    """
    Generate a Reddit sponsered ad for a company based on its sentiment and related news.

    Parameters:
    ----------
    company : str
        The name of the company.
    sentiment : str
        The sentiment of the company (e.g., positive, negative, neutral).
    news_title : str
        The title of the related news article.
    news_summary : str
        A brief summary of the related news article.

    Returns:
    -------
    str
        A generated Reddit sponsered ad.
    """
    try:
        prompt = (
            f"Write a social media post for the company '{company}' based on the following sentiment and news:\n\n"
            f"Sentiment: {sentiment}\n"
            f"News Title: {news_title}\n"
            f"News Summary: {news_summary}\n\n"
            f"The post should be engaging, concise, and relevant to the news."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use a suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates social media posts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating post for {company}: {e}")
        return "Error generating post."
