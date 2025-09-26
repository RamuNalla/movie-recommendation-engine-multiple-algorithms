import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class RecommendationMetrics:        # Comprehensive evaluation metrics for recommendation systems. Includes accuracy, ranking, diversity, and coverage metrics

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def precision_at_k(y_true: List, y_pred: List, k: int, threshold: float = 3.5) -> float:        # Precision at K - fraction of recommended items that are relevant
        
        if len(y_pred) < k:
            k = len(y_pred)
        
        if k == 0:
            return 0.0
        
        top_k_true = y_true[:k]     # Get top k predictions
        relevant_items = sum(1 for rating in top_k_true if rating >= threshold)     # Count relevant items (above threshold). threshold: Threshold for considering an item as relevant
        
        return relevant_items / k
    

    @staticmethod
    def recall_at_k(y_true: List, y_pred: List, k: int, threshold: float = 3.5,     # Recall at K - fraction of relevant items that are recommended
                   all_relevant_items: int = None) -> float:
        """
        Recall at K - fraction of relevant items that are recommended
        
        Args:
            y_true: True ratings for recommended items
            y_pred: Predicted ratings for recommended items
            k: Number of recommendations to consider
            threshold: Threshold for considering an item as relevant
            all_relevant_items: Total number of relevant items for the user
        """
        if len(y_pred) < k:
            k = len(y_pred)
        
        if k == 0:
            return 0.0
        
        top_k_true = y_true[:k]         # Get top k predictions
        relevant_recommended = sum(1 for rating in top_k_true if rating >= threshold)       # Count relevant items in recommendations
        
        if all_relevant_items is None:
            all_relevant_items = sum(1 for rating in y_true if rating >= threshold)         # If total relevant items not provided, use all items above threshold
        
        if all_relevant_items == 0:
            return 0.0
        
        return relevant_recommended / all_relevant_items
    
    @staticmethod
    def f1_score_at_k(y_true: List, y_pred: List, k: int, threshold: float = 3.5) -> float:
        """F1 Score at K"""
        precision = RecommendationMetrics.precision_at_k(y_true, y_pred, k, threshold)
        recall = RecommendationMetrics.recall_at_k(y_true, y_pred, k, threshold)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def dcg_at_k(y_true: List, k: int) -> float:
        """
        Discounted Cumulative Gain at K
        Measures the quality of ranking with position discount
        """
        if len(y_true) < k:
            k = len(y_true)
        
        if k == 0:
            return 0.0
        
        dcg = y_true[0]  # First item has no discount
        for i in range(1, k):
            dcg += y_true[i] / np.log2(i + 1)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(y_true: List, y_pred: List, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at K
        DCG normalized by ideal DCG
        """
        if len(y_pred) < k:
            k = len(y_pred)
        
        if k == 0:
            return 0.0
        
        # Calculate DCG for current ranking
        dcg = RecommendationMetrics.dcg_at_k(y_true[:k], k)
        
        # Calculate ideal DCG (sorted by true ratings)
        ideal_true = sorted(y_true, reverse=True)
        idcg = RecommendationMetrics.dcg_at_k(ideal_true[:k], k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mean_reciprocal_rank(y_true: List, threshold: float = 3.5) -> float:
        """
        Mean Reciprocal Rank - position of first relevant item
        
        Args:
            y_true: True ratings in recommendation order
            threshold: Threshold for relevance
        """
        for i, rating in enumerate(y_true):
            if rating >= threshold:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def hit_rate_at_k(y_true: List, k: int, threshold: float = 3.5) -> float:
        """
        Hit Rate at K - whether at least one relevant item is in top K
        
        Args:
            y_true: True ratings for recommended items
            k: Number of recommendations to consider
            threshold: Threshold for relevance
        """
        if len(y_true) < k:
            k = len(y_true)
        
        if k == 0:
            return 0.0
        
        # Check if any item in top k is relevant
        for i in range(k):
            if y_true[i] >= threshold:
                return 1.0
        
        return 0.0
    
    @staticmethod
    def diversity_score(recommendations: List[List[int]], item_similarity_matrix: np.ndarray) -> float:
        """
        Intra-list diversity - average pairwise dissimilarity of recommended items
        
        Args:
            recommendations: List of recommendation lists for each user
            item_similarity_matrix: Item-item similarity matrix
        """
        if len(recommendations) == 0:
            return 0.0
        
        total_diversity = 0
        valid_users = 0
        
        for user_recs in recommendations:
            if len(user_recs) <= 1:
                continue
            
            pairwise_diversity = 0
            pairs = 0
            
            for i in range(len(user_recs)):
                for j in range(i + 1, len(user_recs)):
                    item_i, item_j = user_recs[i], user_recs[j]
                    # Diversity = 1 - similarity
                    diversity = 1 - item_similarity_matrix[item_i, item_j]
                    pairwise_diversity += diversity
                    pairs += 1
            
            if pairs > 0:
                total_diversity += pairwise_diversity / pairs
                valid_users += 1
        
        return total_diversity / valid_users if valid_users > 0 else 0.0
    
    @staticmethod
    def catalog_coverage(recommendations: List[List[int]], n_items: int) -> float:
        """
        Catalog Coverage - fraction of items that appear in recommendations
        
        Args:
            recommendations: List of recommendation lists for each user
            n_items: Total number of items in catalog
        """
        recommended_items = set()
        
        for user_recs in recommendations:
            recommended_items.update(user_recs)
        
        return len(recommended_items) / n_items
    
    @staticmethod
    def novelty_score(recommendations: List[List[int]], item_popularity: Dict[int, float]) -> float:
        """
        Novelty Score - average novelty of recommended items
        Novelty = -log2(popularity)
        
        Args:
            recommendations: List of recommendation lists for each user
            item_popularity: Dictionary mapping item_id to popularity score
        """
        total_novelty = 0
        total_items = 0
        
        for user_recs in recommendations:
            for item_id in user_recs:
                if item_id in item_popularity:
                    popularity = item_popularity[item_id]
                    novelty = -np.log2(popularity) if popularity > 0 else 0
                    total_novelty += novelty
                    total_items += 1
        
        return total_novelty / total_items if total_items > 0 else 0.0