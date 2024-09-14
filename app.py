import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved models and vectorizers
@st.cache_resource
def load_models():
    knn_classifier0 = joblib.load('knn_classifier0.pkl')
    knn_classifier1 = joblib.load('knn_classifier1.pkl')
    tfidf_vectorizer0 = joblib.load('tfidf_vectorizer0.pkl')
    tfidf_vectorizer1 = joblib.load('tfidf_vectorizer1.pkl')
    return knn_classifier0, knn_classifier1, tfidf_vectorizer0, tfidf_vectorizer1

# Load the datasets
@st.cache_data
def load_data():
    df0 = pd.read_csv('places_v7.csv')
    df1 = pd.read_csv('places_v8.csv')
    return df0, df1

# Function to calculate composite score
def calculate_score(row):
    rating = row['rating']
    rating_count = row['user_ratings_total']
    positive_count = row['positive_words']
    negative_count = row['negative_words']
    
    score = (
        0.2 * rating +
        0.2 * np.log1p(rating_count) +
        0.5 * (positive_count / (positive_count + negative_count + 1)) +
        0.1 * np.log1p(positive_count + negative_count)
    )
    return score

# Function to get predictions for a single category
def get_verified_top_2_predictions(category, df, classifier, tfidf_vectorizer):
    category_tfidf = tfidf_vectorizer.transform([category])
    top_10_predictions = classifier.kneighbors(category_tfidf, n_neighbors=10, return_distance=False)[0]
    
    verified_places = []
    for prediction in top_10_predictions:
        place_row = df.iloc[prediction]
        actual_category = place_row['categories']
        
        if category.lower() in actual_category.lower():
            verified_places.append(place_row)
    
    if len(verified_places) > 0:
        verified_df = pd.DataFrame(verified_places)
        verified_df_sorted = verified_df.sort_values('score', ascending=False).head(2)
        return verified_df_sorted['name'].tolist()
    
    return []

# Main Streamlit app
def main():
    st.title("Place Recommender")

    # Load models and data
    knn_classifier0, knn_classifier1, tfidf_vectorizer0, tfidf_vectorizer1 = load_models()
    df0, df1 = load_data()

    # Recalculate the scores
    df0['score'] = df0.apply(calculate_score, axis=1)
    df1['score'] = df1.apply(calculate_score, axis=1)

    # User input
    input_categories = st.text_input("Enter categories (comma-separated):", "wildlife, theater, safaris")

    if st.button("Get Recommendations"):
        category_list = [category.strip() for category in input_categories.split(',')]

        final_places_list = []

        for category in category_list:
            verified_places_0 = get_verified_top_2_predictions(category, df0, knn_classifier0, tfidf_vectorizer0)
            verified_places_1 = get_verified_top_2_predictions(category, df1, knn_classifier1, tfidf_vectorizer1)
            
            final_places = (verified_places_0 + verified_places_1)[:2]
            
            final_places_list.extend(final_places)

        st.subheader("Top 5 Recommended Places:")
        for i, place in enumerate(final_places_list[:5], 1):
            st.write(f"{i}. {place}")

if __name__ == "__main__":
    main()