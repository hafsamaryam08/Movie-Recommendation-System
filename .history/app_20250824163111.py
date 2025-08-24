# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- load preprocessed objects ----------
# For simplicity in this demo we rebuild matrices on app start.
ratings = pd.read_csv("u.data", sep="\t", names=["userId","movieId","rating","timestamp"])
movies  = pd.read_csv("u.item", sep="|", header=None, usecols=[0,1], names=["movieId","title"], encoding="latin-1")
data = ratings.merge(movies, on="movieId")
R = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
user_sim = pd.DataFrame(cosine_similarity(R), index=R.index, columns=R.index)

def recommend_movies_simple(user_id, R, user_sim, top_n=10):
    # same simple method used in notebook
    similar = user_sim[user_id].sort_values(ascending=False).drop(user_id)
    neighbors = similar.head(5).index
    avg = R.loc[neighbors].mean().sort_values(ascending=False)
    watched = R.loc[user_id]
    recs = avg[watched == 0].head(top_n)
    return recs.reset_index().rename(columns={0:'score','title':'movie'})

st.title("MovieLens — User-based Recommender (Demo)")
user = st.selectbox("Choose user", options=R.index.tolist(), index=0)
k = st.slider("Top-K", min_value=1, max_value=20, value=10)

if st.button("Recommend"):
    recs = recommend_movies_simple(user, R, user_sim, top_n=k)
    if recs.empty:
        st.write("No recommendations available.")
    else:
        st.table(recs.rename(columns={0:'score'}))
