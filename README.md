# Product Sentiment Analysis

## Overview
This project analyzes Amazon product reviews to predict sentiment (Positive, Neutral, Negative) using **Logistic Regression with TF-IDF bigrams** and class balancing. The project demonstrates **NLP preprocessing, model interpretability, and deployment via Streamlit**.

## Features
- Text preprocessing: cleaning, tokenization, and bigram extraction
- Class balancing to improve recall for minority classes (Neutral & Negative)
- Model interpretability: identifies top words contributing to each sentiment
- Streamlit web app for live predictions:
  - Single review input
  - Display of class probabilities
  - Display of top contributing words

## Dataset
- Amazon Fine Food Reviews (CSV format)
- Cleaned and preprocessed for training

## Model Performance
- Accuracy: 79.7%
- Negative recall: 0.75
- Neutral recall: 0.66
- Positive recall: 0.82


