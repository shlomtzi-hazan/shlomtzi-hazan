## **"BuzzWatch: Linking Market News to Real-Time Brand Sentiment"**

---

## ⚙️ **App Design Overview**

### 💡 **Objective**
Analyze the relationship between *news events* and *Reddit sentiment* for **Pepsi-Co**, compared to **Coca-Cola** and **Olipop**, highlighting notable changes over time and identifying impactful news.

---

## 🧱 **App Architecture**

Use a modular approach:

### 1. **Data Pipeline**
| Source       | Data Type     | Tool to Use               |
|--------------|----------------|---------------------------|
| Reddit       | Public posts  | Reddit API (via PRAW)     |
| Google News  | Headlines     | Google News API / SERP API or BeautifulSoup |

### 2. **Sentiment Analysis**
- Use **VADER** (for simplicity) or **transformer-based models** (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) for Reddit posts.
- Assign sentiment score (e.g., -1 to 1) per post, aggregated per hour/day.
- Consider calculating:
  - Rolling sentiment average
  - Volume of posts per company

### 3. **Event Detection from News**
- Extract keywords and named entities using **spaCy** or `keyBERT`.
- Match company mentions (Pepsi, Coca-Cola, Olipop).
- Score “event impact” using frequency + sentiment + post spike after news.

### 4. **Storage**
- Store all data in **SQLite**, **Pandas DataFrames**, or a lightweight **JSON** file depending on simplicity.

---

## 📊 **Dashboard Components (Using Streamlit)**

### 1. 📈 **Sentiment Over Time (Line Chart)**
- X-axis: Time (past week, focus on last 24h).
- Y-axis: Sentiment score.
- One line per company.
- Highlight spike or dip regions with markers/tooltips.

### 2. ⚖️ **Sentiment Comparison (Bar or Radar Chart)**
- Compare sentiment scores in the last 24h or 7 days.
- Show mean sentiment, post volume, or standard deviation per brand.

### 3. 📰 **News Event Impact Panel**
- Show **top 3 impactful news events** (last 24h).
- Include:
  - Headline
  - Source (Google News / Bloomberg / etc.)
  - Related company
  - Link to full article
  - Estimated impact score (optional)

### Optional: Filters
- Date range selector.
- Toggle between absolute sentiment and sentiment delta.

---

## 🧪 **Daily Auto Update**
- Use **cron job** or **Streamlit Cloud's scheduler**.
- Pull new Reddit posts and news every 24h.
- Re-calculate sentiment and update dashboard.

---

## 📝 **What to Include in the Final Report**

1. **Problem Statement**: “How do real-time news events affect online sentiment of Pepsi-Co compared to its competitors?”
2. **Data Collection**: Explain APIs, scraping, preprocessing.
3. **App Development**:
   - Tech stack: Python, Streamlit, PRAW/scraping, VADER/BERT.
   - Architecture diagram.
4. **Analysis**:
   - Sentiment scoring logic.
   - Event detection method.
   - How events map to sentiment shifts.
5. **Screenshots & Visualizations**: From Streamlit dashboard.
6. **Insights**:
   - Did Pepsi sentiment rise after a positive article?
   - How did Coca-Cola and Olipop respond in comparison?
7. **AI Usage**: If you use ChatGPT or similar, show prompts & results.
8. **Creativity**: Show originality in linking events + sentiment (goes beyond class content!).

---

## 🧰 **Tech Stack Suggestions**

| Component              | Tools / Libraries                         |
|------------------------|-------------------------------------------|
| App framework          | Streamlit                                 |
| Reddit scraping        | PRAW                                      |
| News scraping          | Google News API / NewsAPI / BeautifulSoup |
| NLP & Sentiment        | VADER / BERT / TextBlob / spaCy           |
| Visualization          | Plotly / Altair / Matplotlib              |
| Scheduler (for daily)  | `cron`, `apscheduler`, or Streamlit Cloud |
| Optional storage       | SQLite / Firebase / JSON / CSV            |

