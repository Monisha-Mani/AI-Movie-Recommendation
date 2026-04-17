# Import required libraries for preprocessing
import pandas as pd

def preprocess_data():

    # Load datasets
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    # Merge datasets on movieId
    df = pd.merge(ratings, movies, on="movieId")

    # Keep only required columns
    df = df[['userId', 'movieId', 'title', 'genres', 'rating']]

    # Handle missing values
    df['genres'] = df['genres'].fillna("")

    # Save processed data
    df.to_csv("data/processed_data.csv", index=False)

    print("Preprocessing completed. File saved as processed_data.csv")

if __name__ == "__main__":
    preprocess_data()