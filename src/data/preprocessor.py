import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Optional

class DataPreprocessor:                     # Preprocessor for recommendation system data

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def create_user_item_matrix(self, ratings: pd.DataFrame) -> Tuple[np.ndarray, Dict]:      # create user-item rating matrix

        ratings = ratings.copy()                    
        ratings['user_idx'] = self.user_encoder.fit_transform(ratings['user_id'])       # encode users and items to continuous indices
        ratings['item_idx'] = self.item_encoder.fit_transform(ratings["item_id"])

        n_users = len(self.user_encoder.classes_)           # create matrix dimensions
        n_items = len(self.item_encoder.classes_)

        user_item_matrix = np.zeros((n_users, n_items))

        for _, row in ratings.iterrows():
            user_item_matrix[int(row['user_idx']), int(row['item_idx'])] = row['rating']

        matrix_info = {
            'n_users': n_users,
            'n_items': n_items,
            'n_ratings': len(ratings),
            'sparsity': 1 - (len(ratings) / (n_users * n_items)),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder
        }

        self.is_fitted = True
        return user_item_matrix, matrix_info
    
    def create_sparse_matrix(self, ratings: pd.DataFrame) -> csr_matrix:        # create sparse user-item matrix for memory efficiency

        ratings = ratings.copy()

        if not self.is_fitted:
            ratings['user_idx'] = self.user_encoder.fit_transform(ratings['user_id'])
            ratings['item_idx'] = self.item_encoder.fit_transform(ratings['item_id'])
            self.is_fitted = True
        else:
            ratings['user_idx'] = self.user_encoder.transform(ratings['user_id'])
            ratings['item_idx'] = self.item_encoder.transform(ratings['item_id'])

        sparse_matrix = csr_matrix(
            (ratings['rating'].values, 
             (ratings['user_idx'].values, ratings['item_idx'].values)),
            shape=(len(self.user_encoder.classes_), len(self.item_encoder.classes_))
        )

        return sparse_matrix

    def prepare_implicit_feedback(self, ratings: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:     # convert explicit rating to implicit feedback

        implicit_ratings = ratings.copy()
        implicit_ratings['feedback'] = (implicit_ratings['rating'] >= threshold).astype(int)
        return implicit_ratings
    
    def create_content_features(self, movies: pd.DataFrame) -> np.ndarray:      # create content-based features from movie metadata

        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                     'Sci-Fi', 'Thriller', 'War', 'Western']
        
        content_features = movies[genre_cols].values
        return content_features
    
    def create_user_features(self, users: pd.DataFrame) -> np.ndarray:      # create user demographic features (2d numpy array)

        users_encoded = users.copy()

        gender_encoder = LabelEncoder()
        occupation_encoder = LabelEncoder()

        users_encoded['gender_encoded'] = gender_encoder.fit_transform(users_encoded['gender'])
        users_encoded['occupation_encoded'] = occupation_encoder.fit_transform(users_encoded['occupation'])

        users_encoded['age_normalized'] = self.scaler.fit_transform(users_encoded[['age']])

        feature_cols = ['age_normalized', 'gender_encoded', 'occupation_encoded']
        user_features = users_encoded[feature_cols].values

        return user_features 


    def temporal_split(self, ratings: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:       # split data based on timestamp (for more realistic evaluation)

        ratings_sorted = ratings.sort_values('timestamp')
        split_idx = int(len(ratings_sorted) * (1 - test_size))      # find the split index after sorting by timestamp

        train_data = ratings_sorted.iloc[:split_idx]
        test_data = ratings_sorted.iloc[split_idx:]
        
        return train_data, test_data
    
    def get_user_item_mappings(self):           # get mappings between original IDs and encoded indices

        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet. Call create_user_item_matrix first.")
            
        user_mapping = {original: encoded for encoded, original in enumerate(self.user_encoder.classes_)}
        item_mapping = {original: encoded for encoded, original in enumerate(self.item_encoder.classes_)}
        
        return user_mapping, item_mapping



    




        
        












