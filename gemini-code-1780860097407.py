import streamlit as st
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Setup page config
st.set_page_config(page_title="AI Market Researcher", page_icon="📈", layout="wide")

# Initialize VADER Sentiment Engine
@st.cache_resource
def load_sentiment_analyzer():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

# App UI Header
st.title("📈 AI Market Researcher Assistant")
st.markdown("This assistant scans real-time market news and extracts underlying sentiment polarity scores.")

# Sidebar Controls
st.sidebar.header("Control Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper().strip()
max_articles = st.sidebar.slider("Articles to Scan", min_value=3, max_value=15, value=5)

if st.sidebar.button("Run Market Analysis", type="primary"):
    if not ticker:
        st.error("Please enter a valid stock symbol.")
    else:
        with st.spinner(f"Gathering data and analyzing news for {ticker}..."):
            # 1. Fetch data from yfinance
            asset = yf.Ticker(ticker)
            news_items = asset.news
            
            if not news_items:
                st.warning(f"No recent news articles found for ticker symbol: {ticker}")
            else:
                headlines = []
                for item in news_items[:max_articles]:
                    title = item.get('title', '')
                    summary = item.get('summary', '')
                    headlines.append(f"{title}. {summary}")
                
                # 2. Process Sentiment
                total_compound = 0
                breakdown = []
                
                for text in headlines:
                    score = sia.polarity_scores(text)
                    total_compound += score['compound']
                    breakdown.append({"text": text, "score": score['compound']})
                
                avg_score = total_compound / len(headlines)
                
                # 3. Define metrics and colors
                if avg_score >= 0.05:
                    tone, color, alert_type = "BULLISH 📈", "green", "success"
                elif avg_score <= -0.05:
                    tone, color, alert_type = "BEARISH 📉", "red", "error"
                else:
                    tone, color, alert_type = "NEUTRAL ⚖️", "gray", "info"
                
                # Display Results in UI
                st.subheader(f"Analysis Summary for {ticker}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Market Sentiment Tone", tone)
                col2.metric("Average Score (-1 to +1)", f"{avg_score:.4f}")
                col3.metric("Articles Processed", len(headlines))
                
                st.markdown("---")
                st.subheader("Extracted News & Contextual Scores")
                
                for idx, item in enumerate(breakdown, 1):
                    with st.expander(f"Article {idx} | Score: {item['score']:.2f}"):
                        st.write(item['text'])
else:
    st.info("← Enter a stock ticker in the sidebar and click **Run Market Analysis**.")