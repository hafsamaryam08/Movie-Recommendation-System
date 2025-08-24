import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------- Load Data ----------
ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv("u.item", sep="|", encoding="latin-1", 
                     names=["movieId", "title"] + [str(i) for i in range(22)], 
                     usecols=[0,1])

# ---------- Prepare User-Item Matrix ----------
user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# ---------- Item-based Collaborative Filtering ----------
item_matrix = user_item_matrix.T
similarity = cosine_similarity(item_matrix)
item_similarity = pd.DataFrame(similarity, index=item_matrix.index, columns=item_matrix.index)

def recommend_item(movie_id, top_n=5):
    if movie_id not in item_similarity.columns:
        return ["Movie not found!"]
    similar_scores = item_similarity[movie_id].sort_values(ascending=False)[1:top_n+1]
    return movies[movies["movieId"].isin(similar_scores.index)]["title"].tolist()

# ---------- Matrix Factorization (SVD) ----------
svd = TruncatedSVD(n_components=20)
latent_matrix = svd.fit_transform(user_item_matrix)
latent_df = pd.DataFrame(latent_matrix, index=user_item_matrix.index)

def recommend_svd(user_id, top_n=5):
    user_ratings = user_item_matrix.loc[user_id]
    user_pred = np.dot(latent_df.loc[user_id], svd.components_)
    unrated = user_ratings[user_ratings == 0].index
    scores = pd.Series(user_pred, index=user_item_matrix.columns)
    top_movies = scores.loc[unrated].sort_values(ascending=False)[:top_n]
    return movies[movies["movieId"].isin(top_movies.index)]["title"].tolist()

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎥", layout="centered")

# Dark theme with custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    h1, h2, h3 {
        color: #BB86FC;
    }
    .stButton>button {
        background-color: #1F1B24;
        color: #E0E0E0;
        border-radius: 10px;
        border: 1px solid #BB86FC;
    }
    .stSelectbox label {
        color: #BB86FC !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Dark Themed Movie Recommender")

# Select Movie for Item-Based CF
movie_choice = st.selectbox("Pick a Movie:", movies["title"].tolist())
if st.button("Recommend Similar Movies"):
    movie_id = movies[movies["title"] == movie_choice]["movieId"].values[0]
    recs = recommend_item(movie_id)
    st.subheader("Recommended Movies (Item-Based):")
    for r in recs:
        st.write("👉", r)

# Select User for SVD
user_choice = st.number_input("Enter User ID (1–943):", min_value=1, max_value=943, step=1)
if st.button("Recommend for User (SVD)"):
    recs = recommend_svd(user_choice)
    st.subheader("Recommended Movies (SVD):")
    for r in recs:
        st.write("⭐", r)
