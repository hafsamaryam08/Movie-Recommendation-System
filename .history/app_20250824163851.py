import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# ---------------------------
# 🔹 1. Page Config (MUST be first)
# ---------------------------
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 🔹 2. Dark Theme Styling
# ---------------------------
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stApp {
            background-color: #0e1117;
        }
        .stSelectbox label, .stRadio label {
            color: #fafafa !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# 🔹 3. Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# ---------------------------
# 🔹 4. Content-Based Filtering
# ---------------------------
def content_based_recommend(movie_title, top_n=5):
    movie_titles = movies['title'].tolist()
    similarity = cosine_similarity(np.identity(len(movie_titles)))
    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended = [movie_titles[i[0]] for i in scores[1:top_n+1]]
    return recommended

# ---------------------------
# 🔹 5. Collaborative Filtering
# ---------------------------
def collaborative_filtering(userId, top_n=5):
    user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating')
    similarity = cosine_similarity(user_movie_ratings.fillna(0))
    user_sim = similarity[userId - 1]  # adjust for index
    similar_users = np.argsort(user_sim)[::-1][1:top_n+1]
    return similar_users + 1

# ---------------------------
# 🔹 6. Matrix Factorization (SVD)
# ---------------------------
def svd_recommend(userId, top_n=5):
    user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    R = user_movie_ratings.to_numpy()
    u, s, vt = svds(R, k=20)
    s_diag = np.diag(s)
    pred = np.dot(np.dot(u, s_diag), vt)
    user_row = pred[userId - 1]
    top_movies = np.argsort(user_row)[::-1][:top_n]
    return [movies[movies['movieId'] == user_movie_ratings.columns[i]]['title'].values[0] for i in top_movies]

# ---------------------------
# 🔹 7. Streamlit Sidebar Navigation
# ---------------------------
st.sidebar.title("⚙️ Settings")
method = st.sidebar.radio("Select Recommendation Method", ["Content-Based", "Collaborative", "Matrix Factorization (SVD)"])

# ---------------------------
# 🔹 8. Main App Logic
# ---------------------------
st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations using different ML techniques.")

if method == "Content-Based":
    st.subheader("🎥 Content-Based Recommendation")
    movie_choice = st.selectbox("Choose a movie:", movies['title'].values)
    if st.button("Recommend"):
        recs = content_based_recommend(movie_choice)
        st.success("Recommended Movies:")
        for m in recs:
            st.write(f"- {m}")

elif method == "Collaborative":
    st.subheader("👥 Collaborative Filtering")
    user_choice = st.number_input("Enter User ID:", min_value=1, max_value=ratings['userId'].max(), step=1)
    if st.button("Recommend"):
        users = collaborative_filtering(user_choice)
        st.success(f"Users similar to User {user_choice}:")
        for u in users:
            st.write(f"- User {u}")

elif method == "Matrix Factorization (SVD)":
    st.subheader("📊 Matrix Factorization (SVD)")
    user_choice = st.number_input("Enter User ID:", min_value=1, max_value=ratings['userId'].max(), step=1)
    if st.button("Recommend"):
        recs = svd_recommend(user_choice)
        st.success("Recommended Movies:")
        for m in recs:
            st.write(f"- {m}")
