import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')

# Load dataset
df = pd.read_csv("dataset.csv")

# Add sentiment score
analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["headline"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["headline"], df["label"], test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=2000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "market_mood_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("✅ Model trained and saved!")
