# VinoMatch Project
# Streamlit Application: app.py

# --- 1. Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import json
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.preprocessing import MinMaxScaler

# --- 2. Load Data and Model Components ---
# Goal: Load all the pre-trained components and data needed for the app.
# This is done once when the app starts to be efficient.

@st.cache_data
def load_data():
    """
    Loads all necessary data and model components from disk.
    The st.cache_data decorator ensures this function only runs once.
    """
    # Load the main training dataframe
    train_df = pd.read_csv('train_processed.csv')
    
    # Ensure NLTK data is downloaded (for text preprocessing)
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

    # --- Generate model components on the fly ---
    # This replaces loading pre-computed .npy files
    
    # 1. Create and fit the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_train_vectors = vectorizer.fit_transform(train_df['corpus'].fillna(''))
    
    # 2. Create and fit the numerical Scaler
    scaler = MinMaxScaler()
    numerical_features = ['points', 'price', 'value_score']
    train_df['value_score'] = train_df['value_score'].fillna(0.5)
    train_numerical_scaled = scaler.fit_transform(train_df[numerical_features])

    # Create a dictionary for country flag emojis
    country_flags = {
        'US': '🇺🇸', 'France': '🇫🇷', 'Italy': '🇮🇹', 'Spain': '🇪🇸',
        'Portugal': '🇵�', 'Chile': '🇨🇱', 'Argentina': '🇦🇷',
        'Austria': '🇦🇹', 'Australia': '🇦🇺', 'Germany': '🇩🇪'
    }
    
    return train_df, vectorizer, scaler, tfidf_train_vectors, train_numerical_scaled, country_flags

train_df, tfidf_vectorizer, scaler, tfidf_train_vectors, train_numerical_scaled, country_flags = load_data()


# --- 3. Helper Functions ---
# Goal: Define functions to handle recommendation logic.

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

def generate_steward_summary(user_input, recommendations_df):
    recommendations_str = ""
    for index, row in recommendations_df.head(5).iterrows():
        recommendations_str += f"- {row['title']} ({row['variety']})\n"

    is_title = user_input in train_df['title'].values
    prompt_intro = f"A user enjoyed the wine '{user_input}'." if is_title else f"A user is looking for a wine they described as: '{user_input}'."

    prompt = (
        "You are an expert wine steward. {intro} "
        "I have found 5 excellent recommendations for them. "
        "Please write a single, friendly paragraph that summarizes why these wines are a good match. "
        "Use markdown bolding (`**word**`) to highlight key wine characteristics (like **cherry**, **full-bodied**, or **tannins**). "
        "Explain the common themes that link them to the user's original request. "
        "Also, briefly mention something interesting or slightly different about one or two of the recommendations to encourage discovery. "
        "Do not list the wines again."
    ).format(intro=prompt_intro)

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chatHistory}
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if (result.get('candidates') and result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts')):
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return "Sorry, I couldn't generate a summary right now."
    except Exception as e:
        return f"Sorry, an error occurred while generating a summary: {e}"

def get_shared_keywords(original_corpus, recommended_corpus, vectorizer):
    """
    Finds the shared top keywords between two documents based on TF-IDF scores.
    """
    original_words = set(original_corpus.split())
    feature_names = np.array(vectorizer.get_feature_names_out())
    idf_scores = vectorizer.idf_
    word_idf_dict = dict(zip(feature_names, idf_scores))
    
    shared_words = [word for word in recommended_corpus.split() if word in original_words and word in word_idf_dict]
    shared_words.sort(key=lambda word: word_idf_dict.get(word, 0))
    return shared_words[-5:] # Return top 5 most important


def get_recommendations_from_text(user_query, top_n=20):
    processed_query = preprocess_text(user_query)
    query_vector = tfidf_vectorizer.transform([processed_query])
    text_sim_scores = cosine_similarity(query_vector, tfidf_train_vectors).flatten()
    sim_scores = list(enumerate(text_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    wine_indices = [i[0] for i in sim_scores[:top_n]]
    return train_df.iloc[wine_indices]


def get_hybrid_recommendations(wine_title, top_n=20):
    try:
        wine_loc = train_df.index.get_loc(train_df[train_df['title'] == wine_title].index[0])
    except IndexError:
        return None

    text_sim_scores = cosine_similarity(tfidf_train_vectors[wine_loc].reshape(1, -1), tfidf_train_vectors)[0]
    wine_numerical = train_numerical_scaled[wine_loc].reshape(1, -1)
    points_sim = 1 - manhattan_distances(wine_numerical[:, 0].reshape(-1, 1), train_numerical_scaled[:, 0].reshape(-1, 1)).flatten()
    value_sim = 1 - manhattan_distances(wine_numerical[:, 2].reshape(-1, 1), train_numerical_scaled[:, 2].reshape(-1, 1)).flatten()
    
    weights = {'text': 0.7, 'points': 0.15, 'value': 0.15}
    hybrid_sim_scores = (weights['text'] * text_sim_scores + weights['points'] * points_sim + weights['value'] * value_sim)
    
    sim_scores = list(enumerate(hybrid_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    wine_indices = [i[0] for i in sim_scores]
    return train_df.iloc[wine_indices]


# --- 4. Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("VinoMatch 🍷")
st.write("Your personal AI-powered wine steward. Find your next favorite wine!")

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
    st.session_state.summary = None
    st.session_state.user_input = None
    st.session_state.selected_wine = "- Select a wine you like -"
    st.session_state.user_query = ""
    st.session_state.num_to_show = 5

def reset_app_state():
    st.session_state.recommendations = None
    st.session_state.summary = None
    st.session_state.user_input = None
    st.session_state.selected_wine = "- Select a wine you like -"
    st.session_state.user_query = ""
    st.session_state.num_to_show = 5

# --- Main Page Layout ---

# Create a list of wine titles for the dropdown
wine_list = ["- Select a wine you like -"] + sorted(train_df['title'].tolist())

# User inputs, controlled by session state
st.selectbox("Search by a wine you've enjoyed:", options=wine_list, key='selected_wine')
st.text_input("Or, describe the kind of wine you're looking for:", placeholder="e.g., a full-bodied red with notes of cherry and oak", key='user_query')

# Buttons in columns
col1, col2 = st.columns([2, 10])
with col1:
    find_button = st.button("Find Recommendations", type="primary")
with col2:
    # Use the on_click callback to reset the state
    reset_button = st.button("Reset Search", on_click=reset_app_state)

if find_button:
    st.session_state.num_to_show = 5 # Reset display count on new search
    # Prioritize text input if the user has typed something
    if st.session_state.user_query:
        st.session_state.user_input = st.session_state.user_query
        with st.spinner('Searching for wines based on your description...'):
            st.session_state.recommendations = get_recommendations_from_text(st.session_state.user_query)
            st.session_state.summary = generate_steward_summary(st.session_state.user_query, st.session_state.recommendations)
    # If text input is empty, use the dropdown selection
    elif st.session_state.selected_wine != "- Select a wine you like -":
        st.session_state.user_input = st.session_state.selected_wine
        with st.spinner('Finding the perfect matches for you...'):
            st.session_state.recommendations = get_hybrid_recommendations(st.session_state.selected_wine)
            if st.session_state.recommendations is not None:
                st.session_state.summary = generate_steward_summary(st.session_state.selected_wine, st.session_state.recommendations)
    else:
        st.warning("Please select a wine or describe what you're looking for.")
        st.session_state.recommendations = None
        st.session_state.summary = None

# --- Results Display ---
if st.session_state.recommendations is not None:
    
    # --- Sidebar for Filters ---
    with st.sidebar:
        st.header("Filter Your Results")
        
        available_varieties = ["All"] + sorted(st.session_state.recommendations['variety'].unique().tolist())
        available_countries = ["All"] + sorted(st.session_state.recommendations['country'].unique().tolist())
        
        filter_variety = st.selectbox("Variety:", options=available_varieties)
        
        country_options = ["All"] + [f"{country} {country_flags.get(country, '')}" for country in available_countries if country != "All"]
        selected_country_with_flag = st.selectbox("Country:", options=country_options)
        filter_country = selected_country_with_flag.split(' ')[0] if selected_country_with_flag != "All" else "All"

        price_bins = {'All': (0, 10000), '$0 - $20': (0, 20), '$21 - $50': (21, 50), '$51 - $100': (51, 100), '$101+': (101, 10000)}
        filter_price = st.selectbox("Price Range:", options=list(price_bins.keys()))
        
        points_bins = {'All': (0, 101), '94-100 (Superb)': (94, 100), '90-93 (Excellent)': (90, 93), '87-89 (Very Good)': (87, 89), '83-86 (Good)': (83, 86), '80-82 (Acceptable)': (80, 82)}
        filter_points = st.selectbox("Points Range:", options=list(points_bins.keys()))

        # Apply filters
        filtered_recs = st.session_state.recommendations.copy()
        if filter_variety != "All":
            filtered_recs = filtered_recs[filtered_recs['variety'] == filter_variety]
        if filter_country != "All":
            filtered_recs = filtered_recs[filtered_recs['country'] == filter_country]
        
        price_range = price_bins[filter_price]
        points_range = points_bins[filter_points]
        filtered_recs = filtered_recs[
            (filtered_recs['price'] >= price_range[0]) & 
            (filtered_recs['price'] <= price_range[1]) & 
            (filtered_recs['points'] >= points_range[0]) &
            (filtered_recs['points'] <= points_range[1])
        ]

    # --- Main Results Area ---
    st.divider()

    if st.session_state.summary:
        st.markdown(f"##### 🤵 VinoMatch Steward Says...")
        st.markdown(f"> {st.session_state.summary}")
    
    st.header("Your Top Recommendations")

    if not filtered_recs.empty:
        # Get the original wine/query corpus for explainability
        is_title = st.session_state.user_input in train_df['title'].values
        if is_title:
            original_corpus = train_df[train_df['title'] == st.session_state.user_input]['corpus'].iloc[0]
        else:
            original_corpus = preprocess_text(st.session_state.user_input)

        for i, row in filtered_recs.head(st.session_state.num_to_show).iterrows():
            st.subheader(row['title'])
            country_flag = country_flags.get(row['country'], '')
            st.markdown(f"*{row['variety']} from {row['province']}, {row['country']} {country_flag}*")
            st.markdown(f"**🏅 {row['points']} pts** |  **💲 ${row['price']:.2f}**")
            taster = f" (Review by {row['taster_name']})" if row['taster_name'] != "Unknown" else ""
            st.write(f"*{row['description']}*{taster}")
            
            with st.expander("Why was this recommended?"):
                shared_terms = get_shared_keywords(original_corpus, row['corpus'], tfidf_vectorizer)
                st.write("This wine was recommended because it shares these key characteristics with your search:")
                st.write(f"**Shared Key Terms:** `{'`, `'.join(shared_terms)}`")

            st.divider()
    else:
        st.warning("No recommendations match your filter criteria.")

    # "View More" button logic
    if st.session_state.num_to_show < len(filtered_recs):
        if st.button("View More"):
            st.session_state.num_to_show += 5
            st.rerun()