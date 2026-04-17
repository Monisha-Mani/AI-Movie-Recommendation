# Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

def train_model():

    # Load processed dataset
    df = pd.read_csv("data/processed_data.csv")

    # Create user-item matrix
    user_matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

    # SVD (Collaborative Filtering)
    svd = TruncatedSVD(n_components=20)
    matrix_reduced = svd.fit_transform(user_matrix)
    collaborative_pred = np.dot(matrix_reduced, svd.components_)

    # Content-Based (TF-IDF on genres) Align content data with user_matrix movies ONLY
    movie_df = df[['movieId', 'title', 'genres']].drop_duplicates()
    movie_df = movie_df[movie_df['movieId'].isin(user_matrix.columns)]

    # Reset index to align positions
    movie_df = movie_df.reset_index(drop=True)

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_df['genres'])

    # Similarity
    content_similarity = cosine_similarity(tfidf_matrix)

    # Now shapes match
    content_pred = user_matrix.values.dot(content_similarity)

    # Normalize
    content_pred = content_pred / (np.max(content_pred) + 1e-5)

    # Normalize content predictions
    content_pred = content_pred / (np.max(content_pred) + 1e-5)

    # Hybrid combination
    hybrid_pred = 0.7 * collaborative_pred + 0.3 * content_pred

    # Save everything
    with open("model.pkl", "wb") as f:
        pickle.dump(hybrid_pred, f)

    with open("user_index.pkl", "wb") as f:
        pickle.dump(user_matrix.index, f)

    with open("movie_index.pkl", "wb") as f:
        pickle.dump(user_matrix.columns, f)

    movie_df.to_csv("data/movie_lookup.csv", index=False)

    print("Hybrid model trained successfully")

if __name__ == "__main__":
    train_model()