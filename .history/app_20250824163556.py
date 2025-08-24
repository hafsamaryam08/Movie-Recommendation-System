import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ======================
# Load Data
# ======================
@st.cache_data
def load_data():
    ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv("u.item", sep="|", encoding="latin-1",
                         names=["movieId", "title"] + [str(i) for i in range(22)],
                         usecols=[0, 1])
    return ratings, movies

ratings, movies = load_data()

# Merge ratings with movies
data = pd.merge(ratings, movies, on="movieId")

# Create pivot table (user-movie matrix)
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating")

# ======================
# Helper Functions
# ======================
def user_based_cf(user_id, n=5):
    user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
    user_sim_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    # Get similar users
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:6].index

    # Recommend movies from similar users
    sim_users_ratings = user_movie_matrix.loc[similar_users].mean().sort_values(ascending=False)
    watched = user_movie_matrix.loc[user_id].dropna().index
    recommendations = sim_users_ratings.drop(watched).head(n)
    return recommendations.index.tolist()

def item_based_cf(user_id, n=5):
    item_similarity = cosine_similarity(user_movie_matrix.T.fillna(0))
    item_sim_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

    user_ratings = user_movie_matrix.loc[user_id].dropna()
    recommendations = pd.Series(dtype=float)

    for movie, rating in user_ratings.items():
        similar_items = item_sim_df[movie].sort_values(ascending=False)[1:n+1]
        recommendations = recommendations.add(similar_items * rating, fill_value=0)

    recommendations = recommendations.drop(user_ratings.index, errors="ignore")
    return recommendations.sort_values(ascending=False).head(n).index.tolist()

def svd_recommend(user_id, n=5):
    matrix = user_movie_matrix.fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(matrix)

    # Get predicted ratings
    predicted = np.dot(latent_matrix, svd.components_)
    preds_df = pd.DataFrame(predicted, index=matrix.index, columns=matrix.columns)

    user_ratings = user_movie_matrix.loc[user_id].dropna()
    recommendations = preds_df.loc[user_id].drop(user_ratings.index).sort_values(ascending=False).head(n)
    return recommendations.index.tolist()

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide", initial_sidebar_state="expanded")

# Dark Theme CSS
st.markdown("""
    <style>
        body {background-color: #0e1117; color: #fafafa;}
        .stApp {background-color: #0e1117;}
        h1,h2,h3,h4,h5,h6 {color: #00c4ff;}
        .stButton>button {background-color: #262730; color: white; border-radius: 8px;}
        .stButton>button:hover {background-color: #00c4ff; color: black;}
    </style>
""", unsafe_allow_html=True)

st.title("🎥 Movie Recommendation System")
st.write("Get personalized movie suggestions using **Collaborative Filtering & Matrix Factorization (SVD)**.")

# Sidebar
st.sidebar.header("⚙️ Settings")
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=int(user_movie_matrix.index.max()), step=1)
num_recs = st.sidebar.slider("Number of Recommendations", 1, 15, 5)
method = st.sidebar.radio("Select Recommendation Method", ["User-based CF", "Item-based CF", "Matrix Factorization (SVD)"])

# Show data preview
with st.expander("📊 View Dataset Samples"):
    st.write("### Ratings Dataset")
    st.dataframe(ratings.head(10))
    st.write("### Movies Dataset")
    st.dataframe(movies.head(10))

# Generate recommendations
if st.sidebar.button("🔍 Get Recommendations"):
    if method == "User-based CF":
        recs = user_based_cf(user_id, num_recs)
    elif method == "Item-based CF":
        recs = item_based_cf(user_id, num_recs)
    else:
        recs = svd_recommend(user_id, num_recs)

    st.success(f"### 🎯 Recommended Movies for User {user_id}:")
    for i, movie in enumerate(recs, start=1):
        st.write(f"**{i}. {movie}**")
