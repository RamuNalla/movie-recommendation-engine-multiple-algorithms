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






