import streamlit as st
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("📈 AI Market Mood Predictor")
st.write("Predict overall market sentiment (Bullish / Bearish / Neutral) from financial headlines.")

try:
        model = joblib.load("market_mood_model.pkl")
        tfidf = joblib.load("vectorizer.pkl")
except:
        st.warning("⚠️ Model not trained yet. Run train_model.py locally to generate model files.")

headline = st.text_area("Enter financial news headline:")

if st.button("Predict"):
        if headline.strip() == "":
                st.warning("Please enter a headline.")
        else:
                X = tfidf.transform([headline])
                pred = model.predict(X)[0]
                st.success(f"Predicted Market Mood: **{pred}**")
                analyzer = SentimentIntensityAnalyzer()
                score = analyzer.polarity_scores(headline)["compound"]
                st.write(f"Sentiment Score: {score:.3f}")
