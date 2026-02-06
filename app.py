import streamlit as st
import joblib
import numpy as np
import re

# Load model and vectorizer
model = joblib.load('lr_bigram_model.pkl')
vectorizer = joblib.load('tfidf_bigram.pkl')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# App title and description
st.title("Amazon Product Review Sentiment Analysis")
st.write("""
This app predicts the **sentiment** of an Amazon review: Positive, Neutral, or Negative.
It uses a **Logistic Regression model trained with TF-IDF bigrams** and class balancing for accurate results.
""")

# Sidebar options
st.sidebar.header("App Settings")
show_probs = st.sidebar.checkbox("Show prediction probabilities", True)
show_top_words = st.sidebar.checkbox("Show top contributing words", True)

# User input
review_input = st.text_area("Enter the product review below:")

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # Clean and vectorize
        cleaned_review = clean_text(review_input)
        X_input = vectorizer.transform([cleaned_review])

        # Predict
        pred_class = model.predict(X_input)[0]
        pred_probs = model.predict_proba(X_input)[0]

        # Display results
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {pred_class}")

        if show_probs:
            st.subheader("Prediction Probabilities")
            for cls, prob in zip(model.classes_, pred_probs):
                st.write(f"{cls}: {prob:.2f}")

        if show_top_words:
            st.subheader("Top Contributing Words")
            coef = model.coef_[list(model.classes_).index(pred_class)]
            feature_names = np.array(vectorizer.get_feature_names_out())
            top_pos_words = feature_names[np.argsort(coef)[-5:]]
            top_neg_words = feature_names[np.argsort(coef)[:5]]

            st.write("Words pushing sentiment toward this class:", ", ".join(top_pos_words[::-1]))
            st.write("Words pushing sentiment away from this class:", ", ".join(top_neg_words))
