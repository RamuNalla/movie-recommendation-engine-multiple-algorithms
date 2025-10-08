import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:

    def __init__(self, similarity_metric = "cosine", n_neighbors = 50):         # implementation of user-based and Item-based collaborative filering
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.user_similarity = None
        self.item_similarity = None
        self.user_item_matrix = None
        self.user_means = None
    
    def fit(self, user_item_matrix: np.ndarray):        # fit the collaborative filtering model

        self.user_item_matrix = user_item_matrix.copy()

        self.user_means = np.mean(user_item_matrix, axis=1, where=(user_item_matrix != 0))      # calculating the mean for each user
        self.user_means = np.nan_to_num(self.user_means)                # replaces NaN with Zeros (if the user has not rated any items (the entire row is all zeroes))

        self._compute_user_similarity()
        self._compute_item_similarity()

    def _compute_user_similarity(self):             # compute user-user similarity matrix

        if self.similarity_metric == 'cosine':
            
            user_matrix_norm = self.user_item_matrix.copy()
            user_norms = np.linalg.norm(user_matrix_norm, axis=1)
            user_matrix_norm[user_norms == 0] = 1e-10           # handles zero vectors by adding small epsilon (cosine similarity involves division of the norms)         
            
            self.user_similarity = cosine_similarity(user_matrix_norm)      # creates a num_users x num_uses matrix

        elif self.similarity_metric == 'pearson':               # Mean-centered ratings for Pearson correlation
            
            user_matrix_centered = self.user_item_matrix.copy()
            for i in range(len(self.user_means)):
                mask = self.user_item_matrix[i] != 0                    # a boolean array with True for items that user has rated
                user_matrix_centered[i, mask] -= self.user_means[i]     # centering operation
            
            self.user_similarity = np.corrcoef(user_matrix_centered)        # pearson correlation coefficient matrix (1 means perfect positive correlation, -1 means perfect negative correlation)
            self.user_similarity = np.nan_to_num(self.user_similarity)      # replace NaN with zeroes 


    def _compute_item_similarity(self):             # compute item-item similarity
        
        if self.similarity_metric == 'cosine':
            item_matrix = self.user_item_matrix.T                   # Transpose of user-item matrix
            item_norms = np.linalg.norm(item_matrix, axis=1)        # the row represent users
            item_matrix_norm = item_matrix.copy()
            item_matrix_norm[item_norms == 0] = 1e-10
            
            self.item_similarity = cosine_similarity(item_matrix_norm)

        elif self.similarity_metric == 'pearson':
            item_matrix = self.user_item_matrix.T                   # Transpose of user-item matrix             
            item_means = np.mean(item_matrix, axis=1, where=(item_matrix != 0))
            item_means = np.nan_to_num(item_means)
            
            item_matrix_centered = item_matrix.copy()
            for i in range(len(item_means)):
                mask = item_matrix[i] != 0
                item_matrix_centered[i, mask] -= item_means[i]
            
            self.item_similarity = np.corrcoef(item_matrix_centered)
            self.item_similarity = np.nan_to_num(self.item_similarity)

    
    def predict_user_based(self, user_idx: int, item_idx: int) -> float:        # predict rating for specific user and item using user-based collaboratibe filtering

        if self.user_similarity is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        user_similarities = self.user_similarity[user_idx]              # get similar users (row of pre-calculated similarity scores)

        item_raters = np.where(self.user_item_matrix[:, item_idx] != 0)[0]      # find users who rated this item

        if len(item_raters) == 0:
            return self.user_means[user_idx]            # if no one rated for this item, fall back to the user's average rating
        
        relevant_similarities = user_similarities[item_raters]              # filter the similarity scores for user_idx only with who rated for the item
        
        relevant_ratings = self.user_item_matrix[item_raters, item_idx]     # get the actual ratings these users gave to the item

        if len(item_raters) > self.n_neighbors:                 # select top-k neighbors
            top_k_idx = np.argsort(relevant_similarities)[-self.n_neighbors:]
            relevant_similarities = relevant_similarities[top_k_idx]
            relevant_ratings = relevant_ratings[top_k_idx] 

        positive_idx = relevant_similarities > 0                # remove negative similarities for better predictions
        if np.sum(positive_idx) == 0:
            return self.user_means[user_idx]
        
        relevant_similarities = relevant_similarities[positive_idx]
        relevant_ratings = relevant_ratings[positive_idx]

        if np.sum(np.abs(relevant_similarities)) == 0:          
            return self.user_means[user_idx] 
        
        # Weighred average prediction
        prediction = self.user_means[user_idx] + \
                    np.sum(relevant_similarities * (relevant_ratings - self.user_means[item_raters[positive_idx]])) / \
                    np.sum(np.abs(relevant_similarities))

        return np.clip(prediction, 1, 5)            # clip the prediction to a valid rating range
    

    def predict_item_based(self, user_idx: int, item_idx: int) -> float:        # predict rating based on item-based collaborative filtering

        if self.item_similarity is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        item_similarities = self.item_similarity[item_idx]                  # get similar items (row of pre-calculated similarity scores)

        user_items = np.where(self.user_item_matrix[user_idx, :] != 0)[0]    # find items rated by this user

        if len(user_items) == 0:
            return 3.0                  # Global average fallback  

        relevant_similarities = item_similarities[user_items]               # filter the similarity scores for item_idx only with the items rated by the user

        relevant_ratings = self.user_item_matrix[user_idx, user_items]      # get the actual ratings provided by user_idx for these items

        if len(user_items) > self.n_neighbors:          # Select top-k neighbors
            top_k_idx = np.argsort(relevant_similarities)[-self.n_neighbors:]
            relevant_similarities = relevant_similarities[top_k_idx]
            relevant_ratings = relevant_ratings[top_k_idx]

        positive_idx = relevant_similarities > 0        # remove negative similarities
        if np.sum(positive_idx) == 0:
            return 3.0
        
        relevant_similarities = relevant_similarities[positive_idx]
        relevant_ratings = relevant_ratings[positive_idx]

        if np.sum(np.abs(relevant_similarities)) == 0:
            return 3.0
        
        # weighted average prediction (numerator: similarity of each relevant item by the user's rating for that item)
        prediction = np.sum(relevant_similarities * relevant_ratings) / np.sum(np.abs(relevant_similarities))
        
        return np.clip(prediction, 1, 5)


    def recommend_items(self, user_idx: int, n_recommendations: int = 10, method='user_based') -> List[Tuple[int, float]]:      # Recommend top-N items for a user, returns a list of (item_idx, predicted_rating) tuples

        unrated_items = np.where(self.user_item_matrix[user_idx, :] == 0)[0]    # Items not rated by the user

        predictions = []

        for item_idx in unrated_items:
            if method == 'user_based':
                pred = self.predict_user_based(user_idx, item_idx)
            else:
                pred = self.predict_item_based(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)              # sort by predicted rating

        return predictions[:n_recommendations]
    
    def get_user_neighbors(self, user_idx: int, n_neighbors: int = 10) -> List[Tuple[int, float]]:      # get similar users to a given user (returns a list of (neighbor_idx, similarity) tuples)

        if self.user_similarity is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        similarities = self.user_similarity[user_idx]

        similarities[user_idx] = -1             # exclude self

        neighbor_indices = np.argsort(similarities)[-n_neighbors:][::-1]
        neighbors = [(idx, similarities[idx]) for idx in neighbor_indices]

        return neighbors
    
    def item_neighbors(self, item_idx: int, n_neighbors: int = 10) -> List[Tuple[int, float]]:          # get most similar items to a given item

        if self.item_similarity is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        similarities = self.item_similarity[item_idx]

        similarities[item_idx] = -1             # exclude self

        neighbor_indices = np.argsort(similarities)[-n_neighbors:][::-1]
        neighbors = [(idx, similarities[idx]) for idx in neighbor_indices]
        
        return neighbors
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict rating for a user-item pair using the selected similarity metric.
        Defaults to user-based if metric is cosine or pearson, otherwise item-based.
        """
        if self.similarity_metric in ['cosine', 'pearson']:
            return self.predict_user_based(user_idx, item_idx)
        else:
            return self.predict_item_based(user_idx, item_idx)


class MemoryEfficientCF:                # Memory-efficient implementation using sparse matrices and k-NN (better for large datasets)
    # this method never cteas user_num x user_num similarity matrix
    def __init__(self, k=50, similarity_metric = 'cosine'):
        self.k = k
        self.similarity_metric = similarity_metric
        self.user_knn = None
        self.item_knn = None
        self.user_item_matrix = None

    def fit(self, sparse_matrix):       # fit the model using sparse user-item matrix

        self.user_item_matrix = sparse_matrix

        self.user_knn = NearestNeighbors(
            n_neighbors=min(self.k, sparse_matrix.shape[0]-1),
            metric=self.similarity_metric
        )
        self.user_knn.fit(sparse_matrix)

        self.item_knn = NearestNeighbors(
            n_neighbors=min(self.k, sparse_matrix.shape[1]-1),
            metric=self.similarity_metric
        )
        self.item_knn.fit(sparse_matrix.T)

    
    def predict_user_based_sparse(self, user_idx: int, item_idx: int) -> float:     # predict rating usign sparse user-based CF

        distances, indices = self.user_knn.kneighbors(                      # find k nearest neighbors
            self.user_item_matrix[user_idx], n_neighbors=self.k
        )       

        similarities = 1 - distances.flatten()              # convert distances to similarities
        neighbor_indices = indices.flatten()

        neighbor_ratings = []
        neighbor_similarities = []

        for i, neighbor_idx in enumerate(neighbor_indices):
            if neighbor_idx != user_idx:                    # Exclude self
                rating = self.user_item_matrix[neighbor_idx, item_idx]
                if rating > 0:                              # Only consider neighbors who rated the item
                    neighbor_ratings.append(rating)
                    neighbor_similarities.append(similarities[i])

        if not neighbor_ratings:
            return 3.0                  # Default rating
        
        neighbor_ratings = np.array(neighbor_ratings)
        neighbor_similarities = np.array(neighbor_similarities)

        if np.sum(neighbor_similarities) == 0:
            return np.mean(neighbor_ratings)
        
        prediction = np.sum(neighbor_similarities * neighbor_ratings) / np.sum(neighbor_similarities)
        return np.clip(prediction, 1, 5)




    
        













