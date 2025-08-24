import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# Load Data
# -----------------------------
ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv("u.item", sep="|", encoding="latin-1", 
                     names=["movieId", "title"] + [str(i) for i in range(22)], 
                     usecols=[0, 1])

# Create user-item matrix
user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Apply SVD
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(user_item_matrix)
approx_ratings = svd.inverse_transform(matrix_reduced)
approx_ratings_df = pd.DataFrame(approx_ratings, 
                                 index=user_item_matrix.index, 
                                 columns=user_item_matrix.columns)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.write("A simple SVD-based recommender system using MovieLens data.")

# Select user
user_id = st.number_input("Enter User ID:", min_value=1, max_value=int(ratings["userId"].max()), value=1, step=1)

if st.button("Get Recommendations"):
    # Get predicted ratings for this user
    user_ratings = approx_ratings_df.loc[user_id]
    
    # Remove movies already rated
    rated_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    recommendations = user_ratings.drop(rated_movies)
    
    # Get top 10 recommendations
    top_movies = recommendations.sort_values(ascending=False).head(10).index
    top_movies = movies[movies["movieId"].isin(top_movies)]
    
    st.subheader("Top Recommended Movies 🎥")
    for idx, row in top_movies.iterrows():
        st.write(f"- {row['title']}")
