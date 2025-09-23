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


class TFIDFContentRecommender:

    def __init__(self, max_features = 5000, ngram_range = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
        self.tfidf_matrix = None
        self.item_similarity_matrix = None
        self.user_profiles = None

    def _create_content_text(self, movies_df: pd.DataFrame) -> List[str]:       # create text representation of movie content (combines title and genre into a single text)
        
        content_texts = []
        
        for _, movie in movies_df.iterrows():
            
            genre_cols = ['Action', 'Adventure', 'Animation', 'Children',           # Extract genres that are marked as 1
                         'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                         'Sci-Fi', 'Thriller', 'War', 'Western']
            
            genres = [genre for genre in genre_cols if movie[genre] == 1]
            
            title = re.sub(r'\(\d{4}\)', '', movie['title']).strip()                # Clean title (remove year and special characters)
            title = re.sub(r'[^\w\s]', ' ', title)
            
            content_text = title + ' ' + ' '.join(genres)                           # Combine title words and genres
            content_texts.append(content_text)
        
        return content_texts

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):       # fit TF-IDF content-based model
        
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        
    
        content_texts = self._create_content_text(movies_df)            # Create content texts
        
        self.tfidf_matrix = self.tfidf.fit_transform(content_texts)     # Fit TF-IDF vectorizer
        
        self.item_similarity_matrix = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)       # Compute item similarity matrix
        
        self._build_user_profiles_tfidf(ratings_df)         # Build user profiles
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.tfidf.vocabulary_)}")

    
    def _build_user_profiles_tfidf(self, ratings_df: pd.DataFrame):       #  Build user profiles using TF-IDF features
        
        n_users = ratings_df['user_idx'].max() + 1
        n_features = self.tfidf_matrix.shape[1]
        
        self.user_profiles = np.zeros((n_users, n_features))
        
        for user_idx in range(n_users):
            user_ratings = ratings_df[ratings_df['user_idx'] == user_idx]           # find the ratings that user has rated
            
            if len(user_ratings) > 0:
                weights = user_ratings['rating'].values             # Weight by rating
                item_indices = user_ratings['item_idx'].values
                
                rated_item_features = self.tfidf_matrix[item_indices].toarray()     # Get TF-IDF features for rated items
                
                weighted_features = np.average(rated_item_features, axis=0, weights=weights)        # Weighted average
                self.user_profiles[user_idx] = weighted_features

    def predict(self, user_idx: int, item_idx: int) -> float:           # Predict rating using TF-IDF similarity
        
        if self.user_profiles is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        user_profile = self.user_profiles[user_idx].reshape(1, -1)
        item_features = self.tfidf_matrix[item_idx].toarray()
        
        similarity = cosine_similarity(user_profile, item_features)[0, 0]
        predicted_rating = 3.0 + 2.0 * similarity
        
        return np.clip(predicted_rating, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:    # Recommend items using TF-IDF content similarity

        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_feature_importance(self, item_idx: int, top_k: int = 10) -> List[Tuple[str, float]]:        # Get most important TF-IDF features for an item. Returns a List of (feature_word, tfidf_score) tuples
        
        item_vector = self.tfidf_matrix[item_idx].toarray().flatten()
        feature_names = self.tfidf.get_feature_names_out()
        
        top_indices = np.argsort(item_vector)[-top_k:][::-1]        # Get top features
        top_features = [(feature_names[idx], item_vector[idx]) for idx in top_indices if item_vector[idx] > 0]
        
        return top_features


