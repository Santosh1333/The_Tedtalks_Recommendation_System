import numpy as np
import pandas as pd
import nltk
import string  # Adding the import statement for the string module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import streamlit as st
import webbrowser
import datetime
import pytz

def display_time():
    """Displays the current Indian Standard Time."""
    # Set the timezone to Indian Standard Time (IST)
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist)
    current_time = now.strftime("%I:%M:%S %p")
    st.markdown(current_time)


# Set background image
background_image = "your_image.jpg"
# Set CSS for the background
background_css = f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
    }}
    </style>

# Insert background CSS
st.markdown(background_css, unsafe_allow_html=True)

# Your Streamlit app content goes here
st.title('My Streamlit App')
st.write('Welcome to my Streamlit app with a background image!')



if __name__ == "__main__":
  display_time()
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
@st.cache
def recommend_talks_with_sentiment(talk_content, comments, data=df, num_talks=10):
    cos_similarities = get_similarities(talk_content)
    comment_sentiments = comments.apply(analyze_sentiment).values

    # Combine cosine similarities and sentiment analysis
    weighted_score = 0.8 * cos_similarities + (-0.3 )* comment_sentiments
    data['score'] = weighted_score

    # Sort by score and display top recommendations
    recommended_talks = data.sort_values(by='score', ascending=False)
    return recommended_talks[['title', 'publushed_date', 'like_count']].head(num_talks)  # Return only titles

# Define Streamlit app
def main():
    st.title('TED Talk Recommendation System')

    # Input for user to enter their talk content
    talk_content = st.text_input('Enter your talk content:')

    if st.button('Recommend Talks'):
        # Get recommendations
        recommended_titles = recommend_talks_with_sentiment([talk_content], comments)
        
        # Display recommended titles
        st.subheader('Recommended Talks:')
        count = 1  # Initialize count for numbering
        for index, row in recommended_titles.iterrows():
            search_query = row['title'].replace(' ', '+')
            google_link = f"https://www.google.com/search?q={search_query}"
            st.write(f"{count}) {row['title']} - [Go]({google_link})", unsafe_allow_html=True)
            st.write(f"          Published Date: {row['publushed_date']}, Likes: {int(row['like_count'])}")
            count += 1  # Increment count

        # Load more button
        if st.button('Load More'):
            recommended_titles = recommend_talks_with_sentiment([talk_content], comments, num_talks=20)
            for index, row in recommended_titles.iloc[10:].iterrows():
                search_query = row['title'].replace(' ', '+')
                google_link = f"https://www.google.com/search?q={search_query}"
                st.write(f"{count}) {row['title']} - [Go]({google_link})", unsafe_allow_html=True)
                st.write(f"          Published Date: {row['publushed_date']}, Likes: {int(row['like_count'])}")
                count += 1  # Increment count

if __name__ == '__main__':
    main()
