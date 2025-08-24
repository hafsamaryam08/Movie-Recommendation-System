import pandas as pd
from sklearn.decomposition import TruncatedSVD
import streamlit as st

# ===============================
# Load Data
# ===============================
ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv("u.item", sep="|", encoding="latin-1",
                     names=["movieId", "title"] + [str(i) for i in range(22)],
                     usecols=[0, 1])

# Pivot table (user-item matrix)
user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Apply SVD
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(user_item_matrix)
approx_ratings = svd.inverse_transform(matrix_reduced)
approx_ratings_df = pd.DataFrame(approx_ratings,
                                 index=user_item_matrix.index,
                                 columns=user_item_matrix.columns)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# Dark theme styling
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .stSelectbox label, .stMarkdown, .stText, .stHeader {
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #1f1f1f;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #333333;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")
st.write("This app uses **Matrix Factorization (SVD)** on the MovieLens dataset.")

# User selection
user_id = st.selectbox("Choose a User ID:", ratings["userId"].unique())

# Recommend button
if st.button("Get Recommendations"):
    # Predicted ratings for selected user
    user_ratings = approx_ratings_df.loc[user_id]

    # Movies already rated by user
    rated_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()

    # Exclude already rated
    recommendations = user_ratings.drop(rated_movies).sort_values(ascending=False).head(10)

    # Join with movie titles
    recommended_movies = movies[movies["movieId"].isin(recommendations.index)]

    st.subheader("🎯 Top Recommended Movies:")
    for title in recommended_movies["title"].values:
        st.write(f"✅ {title}")
