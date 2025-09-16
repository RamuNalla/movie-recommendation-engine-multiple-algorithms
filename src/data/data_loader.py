import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os

class MovielensDataloader:          # Data loader class for the movielens dataset

    def __init__(self, data_path: str = "data/raw/ml-100k/"):
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.users = None
    
    def load_ratings(self) -> pd.DataFrame:         # load ratings data from u.data file
        ratings_path = os.path.join(self.data_path, "u.data")
        ratings = pd.read_csv(ratings_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"], dtype={'user_id': int, 'item_id': int, 'rating': int, 'timestamp': int})
        
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        self.ratings = ratings        
        
        return self.ratings

    def load_movies(self) -> pd.DataFrame:          # load movies data from u.item file

        movies_path = os.path.join(self.data_path, "u.item")

        movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 
                      'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                      'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                      'Thriller', 'War', 'Western']
        
        movies = pd.read_csv(movies_path, sep='|', names=movie_cols, encoding='latin-1', dtype={'item_id': int})

        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')

        genre_cols = movie_cols[5:]         # Genre columns start from index 5
        movies['genres'] = movies[genre_cols].apply(
            lambda x: [genre for genre, val in zip(genre_cols, x) if val == 1], axis=1
        )
        
        self.movies = movies
        return self.movies


    def load_users(self) -> pd.DataFrame:           # load users data from u.user file

        users_path = os.path.join(self.data_path, "u.user")
        
        users = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            dtype={'user_id': int, 'age': int}
        )

        self.users = users
        return self.users
