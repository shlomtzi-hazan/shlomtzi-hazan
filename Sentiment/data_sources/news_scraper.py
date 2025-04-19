# data_sources/news_scraper.py
from GoogleNews import GoogleNews
import pandas as pd
from datetime import datetime, timedelta

def get_recent_news(companies, hours=24):
    googlenews = GoogleNews(lang='en')
    news_items = []

    for company in companies:
        googlenews.search(company)
        articles = googlenews.result()
        for article in articles:
            pub_date = article.get('date', '')
            if pub_date and 'hour' in pub_date.lower():
                news_items.append({
                    'title': article['title'],
                    'link': article['link'],
                    'published': article['date'],
                    'source': article['media'],
                    'company': company,
                    'url': article['link']
                })

    return pd.DataFrame(news_items)
