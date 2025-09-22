import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Dict, Optional
import re

class ContentBasedRecommender:          # content based filtering using item features

    def __init__(self, similarity_metric = "cosine"):
        self.similarity_metric = similarity_metric
        self.item_similarity_matrix = None
        self.item_features = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.user_profiles = None

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):       # Fit content-based model

        self.movies_df = movies_df.copy()               # dataframe with movie features
        self.ratings_df = ratings_df.copy()             # dataframe with user ratings
        
        self._create_item_features(movies_df)           # Create item features
        
        self._compute_item_similarity()                 # Compute item similarity matrix
        
        self._build_user_profiles(ratings_df)           # Build user profiles


    def _create_item_features(self, movies_df: pd.DataFrame):               # Create feature matrix from movie metadata (horizontally stack tjhe genre freatures and year features)
        
        features = []
        
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',        # Genre features (binary)
                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                     'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genre_features = movies_df[genre_cols].values
        
        year_features = movies_df['year'].fillna(movies_df['year'].median()).values.reshape(-1, 1)      # Year feature (normalized)
        year_features = self.scaler.fit_transform(year_features)
        
        self.item_features = np.hstack([genre_features, year_features])     # Combine features
        self.feature_names = genre_cols + ['year_normalized']
        
        print(f"Created feature matrix with shape: {self.item_features.shape}")

    
    def _compute_item_similarity(self):             # Compute item-item similarity matrix using content features
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(self.item_features)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        

    def _build_user_profiles(self, ratings_df: pd.DataFrame):       # Build user profiles as weighted average of item features (items they have rated)
        
        n_users = ratings_df['user_idx'].max() + 1
        n_features = self.item_features.shape[1]
        
        self.user_profiles = np.zeros((n_users, n_features))
        
        for user_idx in range(n_users):
            user_ratings = ratings_df[ratings_df['user_idx'] == user_idx]       # extracts all ratings given by the current user
            
            if len(user_ratings) > 0:
                # Weight by rating (higher ratings = more influence)
                weights = user_ratings['rating'].values                         # ratings used as weights
                item_indices = user_ratings['item_idx'].values                  # indices of the items the user has rated
                
                rated_item_features = self.item_features[item_indices]              # Get features for rated items
                
                weighted_features = np.average(rated_item_features, axis=0, weights=weights)        # Weighted average of features
                self.user_profiles[user_idx] = weighted_features

    
    def predict(self, user_idx: int, item_idx: int) -> float:      # Predict rating based on content similarity
        
        if self.user_profiles is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Calculate similarity between user profile and item features
        user_profile = self.user_profiles[user_idx].reshape(1, -1)
        item_features = self.item_features[item_idx].reshape(1, -1)
        
        similarity = cosine_similarity(user_profile, item_features)[0, 0]
        
        # Convert similarity to rating scale (1-5)
        # Similarity ranges from -1 to 1, map to 1-5
        predicted_rating = 3.0 + 2.0 * similarity
        
        return np.clip(predicted_rating, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray,          # Recommend items based on content similarity to user's profile
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:
        
        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]             # Get unrated items
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)                          # Sort by predicted rating
        return predictions[:n_recommendations]
    

    def get_similar_items(self, item_idx: int, n_similar: int = 10) -> List[Tuple[int, float]]:     # Get items most similar to a given item, returns a List of (similar_item_idx, similarity_score) tuples
        
        if self.item_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        similarities = self.item_similarity_matrix[item_idx]
        
        similarities[item_idx] = -1                 # Exclude self
        
        similar_indices = np.argsort(similarities)[-n_similar:][::-1]
        similar_items = [(idx, similarities[idx]) for idx in similar_indices]
        
        return similar_items
    

    def explain_recommendation(self, user_idx: int, item_idx: int) -> Dict:     # Provide explanation for why an item was recommended. Returns a Dictionary with explanation details
        
        user_profile = self.user_profiles[user_idx]
        item_features = self.item_features[item_idx]
        
        feature_contributions = user_profile * item_features            # Calculate feature contributions
        
        top_features_idx = np.argsort(np.abs(feature_contributions))[-5:][::-1]     # Get top 5 contributing features
        
        explanation = {
            'predicted_rating': self.predict(user_idx, item_idx),
            'overall_similarity': cosine_similarity(
                user_profile.reshape(1, -1), 
                item_features.reshape(1, -1)
            )[0, 0],
            'top_features': [
                {
                    'feature': self.feature_names[idx],
                    'user_preference': user_profile[idx],
                    'item_value': item_features[idx],
                    'contribution': feature_contributions[idx]
                }
                for idx in top_features_idx
            ]
        }
        
        return explanation

