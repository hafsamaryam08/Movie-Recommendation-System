import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv("u.item", sep="|", encoding="latin-1", 
                         names=["movieId", "title"] + [str(i) for i in range(22)], 
                         usecols=[0,1])
    return ratings, movies

ratings, movies = load_data()

# -----------------------------
# Train SVD Recommender
# -----------------------------
@st.cache_data
def train_svd(ratings):
    user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = svd.fit_transform(user_item_matrix)
    approx_ratings = svd.inverse_transform(matrix_reduced)
    approx_ratings_df = pd.DataFrame(approx_ratings, 
                                     index=user_item_matrix.index, 
                                     columns=user_item_matrix.columns)
    return approx_ratings_df

approx_ratings_df = train_svd(ratings)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System (SVD)")

st.write("Select a user to get personalized movie recommendations.")

# Select User
user_id = st.selectbox("Choose User ID", approx_ratings_df.index)

# Number of recommendations
top_n = st.slider("Number of Recommendations", 1, 10, 5)

# Get predictions for the selected user
user_preds = approx_ratings_df.loc[user_id]

# Remove already rated movies
rated_movies = ratings[ratings.userId == user_id]["movieId"].tolist()
user_unseen = user_preds.drop(rated_movies)

# Get top-N recommendations
top_movies = user_unseen.sort_values(ascending=False).head(top_n).index
recommendations = movies[movies.movieId.isin(top_movies)]

st.subheader(f"Top {top_n} Recommended Movies for User {user_id}")
st.table(recommendations)
