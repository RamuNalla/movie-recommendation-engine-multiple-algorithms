import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from typing import List, Tuple, Dict, Any
from metrics import RecommendationMetrics
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:

    def __init__(self, k_values: List[int] = [5, 10, 20], threshold: float = 3.5):
        self.k_values = k_values
        self.threshold = threshold
        self.metrics = RecommendationMetrics()

    def evaluate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:        # Evaluate rating prediction accuracy
        
        results = {
            'RMSE': self.metrics.rmse(y_true, y_pred),
            'MAE': self.metrics.mae(y_true, y_pred),
            'Correlation': stats.pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
        }
        
        return results
    
    def evaluate_ranking(self, test_data: pd.DataFrame, model, user_item_matrix: np.ndarray) -> Dict[str, float]:   # Evaluate ranking quality using various metrics
        """
        Evaluate ranking quality using various metrics
        
        Args:
            test_data: Test dataset with user_idx, item_idx, rating columns
            model: Trained recommendation model
            user_item_matrix: User-item matrix for excluding training items
            
        Returns:
            Dictionary with ranking metrics
        """
        results = {}
        user_metrics = {f'precision_at_{k}': [] for k in self.k_values}
        user_metrics.update({f'recall_at_{k}': [] for k in self.k_values})
        user_metrics.update({f'ndcg_at_{k}': [] for k in self.k_values})
        user_metrics.update({f'hit_rate_at_{k}': [] for k in self.k_values})
        user_metrics['mrr'] = []
        
        # Group test data by user
        user_groups = test_data.groupby('user_idx')
        
        for user_idx, user_data in user_groups:
            # Get recommendations for this user
            try:
                recommendations = model.recommend_items(user_idx, user_item_matrix, 
                                                       n_recommendations=max(self.k_values))
                
                if not recommendations:
                    continue
                
                # Get true ratings for recommended items
                rec_items = [item_idx for item_idx, _ in recommendations]
                true_ratings = []
                
                for item_idx in rec_items:
                    item_rating = user_data[user_data['item_idx'] == item_idx]['rating']    # Find true rating in test data
                    if len(item_rating) > 0:
                        true_ratings.append(item_rating.iloc[0])
                    else:
                        true_ratings.append(0)  # Not rated = not relevant
                
                # Calculate metrics for each k
                for k in self.k_values:
                    user_metrics[f'precision_at_{k}'].append(
                        self.metrics.precision_at_k(true_ratings, rec_items, k, self.threshold)
                    )
                    user_metrics[f'recall_at_{k}'].append(
                        self.metrics.recall_at_k(true_ratings, rec_items, k, self.threshold)
                    )
                    user_metrics[f'ndcg_at_{k}'].append(
                        self.metrics.ndcg_at_k(true_ratings, rec_items, k)
                    )
                    user_metrics[f'hit_rate_at_{k}'].append(
                        self.metrics.hit_rate_at_k(true_ratings, k, self.threshold)
                    )
                
                # Mean Reciprocal Rank
                user_metrics['mrr'].append(
                    self.metrics.mean_reciprocal_rank(true_ratings, self.threshold)
                )
                
            except Exception as e:
                print(f"Error evaluating user {user_idx}: {e}")
                continue
        
        # Calculate average metrics
        for metric_name, values in user_metrics.items():
            if values:
                results[metric_name] = np.mean(values)
            else:
                results[metric_name] = 0.0
        
        return results
    
    
    def evaluate_diversity_and_coverage(self, test_users: List[int], model,         # Evaluate diversity, coverage, and novelty metrics
                                      user_item_matrix: np.ndarray,
                                      item_similarity_matrix: np.ndarray = None,
                                      item_popularity: Dict[int, float] = None) -> Dict[str, float]:
        """
        Evaluate diversity, coverage, and novelty metrics
        
        Args:
            test_users: List of user indices to evaluate
            model: Trained recommendation model
            user_item_matrix: User-item matrix
            item_similarity_matrix: Item similarity matrix for diversity calculation
            item_popularity: Item popularity scores for novelty calculation
            
        Returns:
            Dictionary with diversity and coverage metrics
        """
        results = {}
        all_recommendations = []
        
        # Generate recommendations for all test users
        for user_idx in test_users:
            try:
                recommendations = model.recommend_items(user_idx, user_item_matrix, 
                                                       n_recommendations=10)
                rec_items = [item_idx for item_idx, _ in recommendations]
                all_recommendations.append(rec_items)
            except:
                all_recommendations.append([])
        
        results['catalog_coverage'] = self.metrics.catalog_coverage(        # Calculate catalog coverage
            all_recommendations, user_item_matrix.shape[1]
        )
        
        if item_similarity_matrix is not None:                          # Calculate diversity if similarity matrix provided
            results['diversity_score'] = self.metrics.diversity_score(
                all_recommendations, item_similarity_matrix
            )
        
        if item_popularity is not None:                             # Calculate novelty if popularity scores provided
            results['novelty_score'] = self.metrics.novelty_score(
                all_recommendations, item_popularity
            )
        
        return results
    
    def cross_validate(self, model_class, model_params: Dict, 
                      user_item_matrix: np.ndarray, ratings_df: pd.DataFrame,
                      cv_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation evaluation
        
        Args:
            model_class: Model class to evaluate
            model_params: Parameters for model initialization
            user_item_matrix: User-item rating matrix
            ratings_df: Ratings DataFrame
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with lists of metrics for each fold
        """
        
        from sklearn.model_selection import KFold
        
        fold_results = {
            'rmse': [], 'mae': [], 'precision_at_10': [], 
            'recall_at_10': [], 'ndcg_at_10': []
        }
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
            print(f"Evaluating fold {fold + 1}/{cv_folds}")
            
            train_data = ratings_df.iloc[train_idx]             # Split data
            test_data = ratings_df.iloc[test_idx]
            
            train_matrix = np.zeros_like(user_item_matrix)      # Create training user-item matrix
            for _, row in train_data.iterrows():
                train_matrix[int(row['user_idx']), int(row['item_idx'])] = row['rating']
            
            model = model_class(**model_params)                 # Train model
            model.fit(train_matrix)
            
            y_true = []                                         # Predict ratings for test set
            y_pred = []
            
            for _, row in test_data.iterrows():
                user_idx = int(row['user_idx'])
                item_idx = int(row['item_idx'])
                true_rating = row['rating']
                
                try:
                    pred_rating = model.predict(user_idx, item_idx)
                    y_true.append(true_rating)
                    y_pred.append(pred_rating)
                except:
                    continue
            
            if len(y_true) > 0:
                fold_results['rmse'].append(self.metrics.rmse(np.array(y_true), np.array(y_pred)))      # Accuracy metrics
                fold_results['mae'].append(self.metrics.mae(np.array(y_true), np.array(y_pred)))
                
                ranking_results = self.evaluate_ranking(test_data, model, train_matrix)                 # Ranking metrics (simplified)
                fold_results['precision_at_10'].append(ranking_results.get('precision_at_10', 0))
                fold_results['recall_at_10'].append(ranking_results.get('recall_at_10', 0))
                fold_results['ndcg_at_10'].append(ranking_results.get('ndcg_at_10', 0))
        
        return fold_results