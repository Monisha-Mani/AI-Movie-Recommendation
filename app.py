# Import required libraries for Flask app
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and artifacts
model = pickle.load(open("model.pkl", "rb"))
user_index = pickle.load(open("user_index.pkl", "rb"))
movie_index = pickle.load(open("movie_index.pkl", "rb"))

# Load datasets
movie_df = pd.read_csv("data/movie_lookup.csv")
df = pd.read_csv("data/processed_data.csv")


# Cold Start: Popular Movies

def get_popular_movies(top_n=5):
    popular = df.groupby('title')['rating'].mean().sort_values(ascending=False)
    return popular.head(top_n).index.tolist()


# Recommendation Function

def recommend_movies(user_id, top_n=5):

    # Handle cold start (unknown user)
    if user_id not in set(user_index):
        return get_popular_movies(top_n)

    # Get user position
    user_pos = list(user_index).index(user_id)
    scores = model[user_pos]

    # Movies already watched
    watched = df[df['userId'] == user_id]['movieId'].values

    # Sort predictions
    sorted_items = np.argsort(scores)[::-1]

    recommended = []

    for idx in sorted_items:
        movie_id = movie_index[idx]

        # Skip already watched movies
        if movie_id not in watched:
            movie_row = movie_df[movie_df['movieId'] == movie_id]

            if not movie_row.empty:
                movie_name = movie_row['title'].values[0]
                recommended.append(movie_name)

        # Limit results
        if len(recommended) == top_n:
            break

    return recommended


# Routes

# ============================
# Routes
# ============================

# Home page
@app.route('/')
def home():
    return render_template("index.html")


# Recommendation route (POST)
@app.route('/recommend', methods=['POST'])
def recommend():

    # Handle invalid input
    try:
        user_id = int(request.form.get('user_id'))
    except:
        return render_template("index.html", error="Invalid input. Please enter a valid number.")

    # NEW: validate user range
    if user_id not in set(user_index):
        return render_template("index.html", error="User not found. Please enter a valid User ID.")

    recommendations = recommend_movies(user_id)
    
    return render_template("index.html", recommendations=recommendations)


# Optional: test route (helps debugging)
@app.route('/test')
def test():
    return "App is working"

# Run App

if __name__ == "__main__":
    app.run(debug=True)