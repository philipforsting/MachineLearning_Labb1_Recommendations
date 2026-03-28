import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import numpy as np



def load_data_files():   
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    tags = pd.read_csv('data/tags.csv')

    movies = movies.drop_duplicates("title")
    movies = movies[movies["genres"] != "(no genres listed)"]

    return movies, ratings, tags

def build_feature_matrix(movies, ratings, tags):

    # Exclude movies with few ratings
    min_rating_per_movie = 50   # Set to zero to disable filtering
    y = ratings.groupby("movieId").count()["rating"] > min_rating_per_movie
    ratings_for_famous_movies = y[y].index
    movies_famous = movies[movies["movieId"].isin(ratings_for_famous_movies)].reset_index(drop=True)
    
    # Make words from movies and genres, then merge and create featurematrix
    movies_famous["genres"] = movies_famous["genres"].str.replace("|", " ", regex=False) # "Adventure|Animation..."" -> "Adventure" "Animation"...
    
    tags_for_famous_movies = tags[tags["movieId"].isin(movies_famous["movieId"])]
    tags_for_famous_movies = tags_for_famous_movies.dropna(subset=["tag"])
    tags_grouped = tags_for_famous_movies.groupby("movieId")["tag"].apply(" ".join)         # All tags for specific movie grouped

    movies_features_df = movies_famous.merge(tags_grouped, on="movieId", how="left").reset_index(drop=True)             # column for grouped tags is added to movies_famous
    movies_features_df["tag"] = movies_features_df["tag"].fillna("")
    movies_features_df["combined_text"] = movies_features_df["genres"] + " " + movies_features_df["tag"]
    movies_features_df = movies_features_df.drop(columns = ["genres", "tag"])
    return movies_features_df

def compute_similarity(matrix):
    tfidf = TfidfVectorizer(stop_words="english")
    feature_matrix = tfidf.fit_transform(matrix["combined_text"])
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix
    
def recommend(movie_title, matrix, model):
    match = matrix[matrix["title"] == movie_title]
    if match.empty:
       return "Movie not found"
    index = match.index[0]

    similar_movies = sorted(list(enumerate(model[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    for idx, similarity in similar_movies:
        item = []
        title = matrix.iloc[idx]["title"]
        similarity_proc = round(float(similarity * 100), 1)      
        item.append(title)
        item.append(similarity_proc)
        data.append(item)
    data_df = pd.DataFrame(data)
    print(tabulate(data_df, headers='keys', tablefmt='psql'))
    return 

def recommend_df(movie_title, matrix, similarity_matrix, n=5):
    match = matrix[matrix["title"] == movie_title]
    if match.empty:
        return "Movie not found"
    idx = match.index[0]
    similarity_df = pd.DataFrame({"similarity_score_%": 100*similarity_matrix[idx]}) # Array from cosine_similarity to pandas dataframe
    similarity_df = similarity_df.sort_values(by="similarity_score_%", ascending=False)
    similarity_df = similarity_df.iloc[1:n+1]   # Removing self

    similarity_df["title"] = matrix.loc[similarity_df.index, "title"]  # Replaces movieId with title 
    similarity_df = similarity_df.loc[:, ['title', 'similarity_score_%']].reset_index(drop=True)         # Changing order of columns
    print(tabulate(similarity_df, headers='keys', tablefmt='psql'))
    return similarity_df

def main(movie_title):
    movies, ratings, tags = load_data_files()
    feature_matrix = build_feature_matrix(movies, ratings, tags)
    similarity_matrix = compute_similarity(feature_matrix)
    #recommend(movie_title, feature_matrix, similarity_matrix)
    recommend_df("Yellow Submarine (1968)", feature_matrix, similarity_matrix)


if __name__ == '__main__':
#    movie_title = input("Input a book name: ")
#    main(movie_title)
    main("Yellow Submarine (1968)")