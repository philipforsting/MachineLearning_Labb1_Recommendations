import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import numpy as np



def load_data_files():   
    """Reads csv files and converts them to dataframes"""
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    tags = pd.read_csv('data/tags.csv')

    movies = movies.drop_duplicates("title")
    movies = movies[movies["genres"] != "(no genres listed)"]

    return movies, ratings, tags

def build_feature_matrix(movies, ratings, tags):
    """Feature matrix is built from movies by excluding impopular movies and combinding tags with popular movies"""
    # Exclude movies with few ratings
    min_rating_per_movie = 50   # Set to zero to disable filtering
    y = ratings.groupby("movieId").count()["rating"] > min_rating_per_movie
    ratings_for_famous_movies = y[y].index
    movies_famous = movies[movies["movieId"].isin(ratings_for_famous_movies)].reset_index(drop=True)
    
    # Make words from movies and genres, then merge and create featur ematrix
    movies_famous["genres"] = movies_famous["genres"].str.replace("|", " ", regex=False) # "Adventure|Animation..."" -> "Adventure" "Animation"...
    
    tags_for_famous_movies = tags[tags["movieId"].isin(movies_famous["movieId"])]
    tags_for_famous_movies = tags_for_famous_movies.dropna(subset=["tag"])
    tags_grouped = tags_for_famous_movies.groupby("movieId")["tag"].apply(" ".join)         # All tags for specific movie grouped

    movies_features_df = movies_famous.merge(tags_grouped, on="movieId", how="left").reset_index(drop=True)  # column for grouped tags is added to movies_famous
    movies_features_df["tag"] = movies_features_df["tag"].fillna("")
    movies_features_df["combined_text"] = movies_features_df["genres"] + " " + movies_features_df["tag"]
    movies_features_df = movies_features_df.drop(columns = ["genres", "tag"])
    return movies_features_df

def compute_similarity(matrix):
    """cosine matrix is computed for all moivies"""
    tfidf = TfidfVectorizer(stop_words="english")
    feature_matrix = tfidf.fit_transform(matrix["combined_text"])
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix
    

def recommend(movie_title, matrix, similarity_matrix,):
    """Extracts the movies simular to the inout movies from the feature matrix and presents it in a printed table in the terminal"""
    match = matrix[matrix["title"] == movie_title]
    if match.empty:
        return "Movie not found"
    idx = match.index[0]
    similarity_df = pd.DataFrame({"similarity_score_%": 100*similarity_matrix[idx]}).round({"similarity_score_%":1})  # Array from cosine_similarity to pandas dataframe
    similarity_df = similarity_df.sort_values(by="similarity_score_%", ascending=False)
    similarity_df = similarity_df.iloc[1:5+1] # Removing self

    similarity_df["title"] = matrix.loc[similarity_df.index, "title"]  # Replaces movieId with title 
    similarity_df = similarity_df.loc[:, ['title', 'similarity_score_%']]  # Changing order of columns
    similarity_df.reset_index(drop=True) 

    print(tabulate(similarity_df, headers='keys', tablefmt='psql')) #Making nice print in terminal
    return similarity_df

def main(movie_title):
    movies, ratings, tags = load_data_files()
    feature_matrix = build_feature_matrix(movies, ratings, tags)
    similarity_matrix = compute_similarity(feature_matrix)
    recommend(movie_title, feature_matrix, similarity_matrix)


if __name__ == '__main__':
    movie_title = input("Input a movie title: ")
    main(movie_title)
# Yellow Submarine (1968)