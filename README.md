# 🎬 Movie Recommendation System (Task 5 - Elevvo ML Internship)

This project implements a **Movie Recommendation System** using **Collaborative Filtering** and **Matrix Factorization (SVD)**.  
It recommends movies to users based on their past ratings and similarities with other users/items.  
Additionally, a **Streamlit-based dark-themed UI** is provided to make recommendations user-friendly and interactive.  

---

## 🚀 Features  
- User-based collaborative filtering (find similar users)  
- Item-based collaborative filtering (find similar movies)  
- Matrix Factorization using **Singular Value Decomposition (SVD)**  
- **Streamlit UI** for interaction  

---

## 🛠️ Technologies Used
- **Python 3**  
- **Pandas** (data manipulation)  
- **NumPy** (numerical computations)  
- **Scikit-learn** (cosine similarity, matrix factorization)  
- **Surprise Library** (SVD implementation for collaborative filtering)  
- **Streamlit** (UI for the app, dark theme)  

---

## 📂 Dataset
The project uses the **MovieLens 100k dataset** (`u.data`, `u.item`):  
- `u.data`: Contains user ratings (`userId`, `movieId`, `rating`, `timestamp`).  
- `u.item`: Contains movie metadata (`movieId`, `title`).  
