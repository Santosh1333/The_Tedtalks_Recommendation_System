import numpy as np
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import pickle
import streamlit as st

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load TED Talks data
df = pd.read_csv('JOINT_ted_video_transcripts_comments_stats.csv')

# Remove rows with null values
df = df.dropna()

# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

# Combine title and transcript, then preprocess
df['details'] = df['title'] + ' ' + df['transcript']
df['details'] = df['details'].apply(preprocess_text)

# Function to perform sentiment analysis on comments
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    return analysis.sentiment.polarity

# Load and preprocess comments
comments = df['comments'].fillna('').astype(str)
comments = comments.apply(preprocess_text)

# Function to calculate similarities between input talk_content and dataset
def get_similarities(talk_content, data=df):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data['details'])
    talk_array1 = vectorizer.transform(talk_content)
    details_array = vectorizer.transform(data['details'])  # Vectorize all details at once
    sim = cosine_similarity(talk_array1, details_array)
    return sim.flatten()  # Flatten the similarity matrix to a 1D array

# Function to recommend talks with sentiment analysis
def recommend_talks_with_sentiment(talk_content, comments, data=df):
    cos_similarities = get_similarities(talk_content)
    comment_sentiments = comments.apply(analyze_sentiment).values

    # Combine cosine similarities and sentiment analysis
    weighted_score = 0.7 * cos_similarities + 0.3 * comment_sentiments
    data['score'] = weighted_score

    # Sort by score and display top recommendations
    recommended_talks = data.sort_values(by='score', ascending=False)
    return recommended_talks['title'].tolist()  # Return only titles

# Define Streamlit app
def main():
    st.title('TED Talk Recommendation System')

    # Input for user to enter their talk content
    talk_content = st.text_input('Enter your talk content:')

    # Get the current position from session state or set it to 0 if not yet initialized
    if 'position' not in st.session_state:
        st.session_state.position = 0

    if st.button('Recommend Talks'):
        # Get recommendations
        recommended_titles = recommend_talks_with_sentiment([talk_content], comments)
        
        # Display recommended titles
        st.subheader('Recommended Talks:')
        for index, title in enumerate(recommended_titles[st.session_state.position : st.session_state.position + 5]):
            st.write(f"{index+1}. {title}")

        if len(recommended_titles) > st.session_state.position + 5:
            # Update position when "Next" button is clicked
            if st.button('Next'):
                st.session_state.position += 5

if __name__ == '__main__':
    main()
