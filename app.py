import streamlit as st
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict

@st.cache_resource
def load_models():
    knn_classifier0 = joblib.load('knn_classifier0.pkl')
    knn_classifier1 = joblib.load('knn_classifier1.pkl')
    tfidf_vectorizer0 = joblib.load('tfidf_vectorizer0.pkl')
    tfidf_vectorizer1 = joblib.load('tfidf_vectorizer1.pkl')
    return knn_classifier0, knn_classifier1, tfidf_vectorizer0, tfidf_vectorizer1

knn_classifier0, knn_classifier1, tfidf_vectorizer0, tfidf_vectorizer1 = load_models()

@st.cache_data
def load_data():
    df0 = pd.read_csv('places_v7.csv')
    df1 = pd.read_csv('places_v8.csv')
    return df0, df1

df0, df1 = load_data()

def calculate_score(row: pd.Series) -> float:
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

df0['score'] = df0.apply(calculate_score, axis=1)
df1['score'] = df1.apply(calculate_score, axis=1)

def get_verified_predictions(category: str, df: pd.DataFrame, classifier, tfidf_vectorizer, n_neighbors: int = 10) -> List[Dict]:
    category_tfidf = tfidf_vectorizer.transform([category])
    top_predictions = classifier.kneighbors(category_tfidf, n_neighbors=n_neighbors, return_distance=False)[0]

    verified_places = []
    for prediction in top_predictions:
        place_row = df.iloc[prediction]
        if category.lower() in place_row['categories'].lower():
            verified_places.append(place_row)
    
    if verified_places:
        verified_df = pd.DataFrame(verified_places)
        verified_df_sorted = verified_df.sort_values('score', ascending=False)
        return verified_df_sorted[['name', 'rating', 'user_ratings_total', 'score']].to_dict('records')
    
    return []

def predict_places(input_categories: str) -> List[str]:
    category_list = [category.strip() for category in input_categories.split(',')]
    final_places_list = []

    for category in category_list:
        verified_places_0 = get_verified_predictions(category, df0, knn_classifier0, tfidf_vectorizer0)
        verified_places_1 = get_verified_predictions(category, df1, knn_classifier1, tfidf_vectorizer1)
        
        combined_places = verified_places_0 + verified_places_1
        if combined_places:
            final_places_list.append(combined_places[0]['name'])

    remaining_slots = 5 - len(final_places_list)
    if remaining_slots > 0:
        all_remaining_places = []
        for category in category_list:
            verified_places_0 = get_verified_predictions(category, df0, knn_classifier0, tfidf_vectorizer0)
            verified_places_1 = get_verified_predictions(category, df1, knn_classifier1, tfidf_vectorizer1)
            all_remaining_places.extend(verified_places_0 + verified_places_1)

        remaining_places_df = pd.DataFrame(all_remaining_places)
        if not remaining_places_df.empty:
            remaining_places_df = remaining_places_df.sort_values('score', ascending=False)
            remaining_places_df = remaining_places_df[~remaining_places_df['name'].isin(final_places_list)]
            final_places_list.extend(remaining_places_df['name'].head(remaining_slots).tolist())

    while len(final_places_list) < 5:
        final_places_list.append(f"No additional place found {len(final_places_list) + 1}")

    return final_places_list[:5]

st.title('Place Predictor')

st.image("https://th.bing.com/th/id/R.18a7ed761e763c783ef54b31d967e37d?rik=tUwLgcZq%2bxlUOw&pid=ImgRaw&r=0", caption="Sri Lankan Tourism", use_column_width=True)

activity_categories = ['cycling', 'historical monuments', 'village homestays', 'butterfly watching', 'hot springs', 'wildlife viewing', 'sea cruises', 'themed parks', 'craft workshops', 'fishing', 'sailing', 'history tours', 'literary tours', 'public art installations', 'temple pilgrimages', 'architecture tours', 'golfing', 'hot air ballooning', 'spiritual retreats', 'cultural experiences', 'botanical gardens', 'boat safaris', 'caving', 'cultural festivals', 'museum visits', 'mountain biking', 'camping', 'turtle watching', 'historic walks', 'safaris', 'waterfalls', 'scuba diving', 'elephant rides', 'bird watching', 'ayurvedic spa treatments', 'horse shows', 'traditional ceremonies', 'surfing', 'historic sites', 'art classes', 'city tours', 'theater', 'amusement parks', 'architecture photography', 'beachfront dining', 'kayaking', 'beach visits', 'rock climbing', 'arts and culture', 'snorkeling', 'animal encounters', 'archaeological sites', 'sailing lessons', 'whale watching', 'local crafts', 'yoga retreats', 'paddleboarding', 'horseback riding', 'zip-lining', 'outdoor adventures', 'planetarium visits', 'water parks', 'photography', 'sightseeing', 'tea tasting', 'hiking', 'river cruises', 'landscape photography']

st.write('Please input from these Activity Categories:')
st.write(', '.join(activity_categories)) 
input_categories = st.text_input("Enter categories (comma-separated):", "wildlife")

if st.button('Predict Places'):
   
    predicted_places = predict_places(input_categories)
    
    
    st.subheader("Top 5 Predicted Places:")
    for i, place in enumerate(predicted_places, 1):
        st.write(f"{i}. {place}")
