import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WeightedHybrid:       # Combines prediction from multiple models using learned weights

    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None): # Initialize weighted hybrid model
        """
        Args:
            models: Dictionary of {model_name: model_instance}
            weights: Dictionary of {model_name: weight}. If None, equal weights are used
        """
        self.models = models
        self.model_names = list(models.keys())
        
        if weights is None:
            self.weights = {name: 1.0 / len(models) for name in self.model_names}       # Equal weights for all models
        else:                                           # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            self.weights = {name: weight / total_weight for name, weight in weights.items()}
        
        self.is_fitted = False

    def fit(self, *args, **kwargs):
        """
        Fit all component models
        Note: Each model should be fitted individually before using this hybrid
        """
        self.is_fitted = True
        print("Weighted Hybrid: Component models should be fitted individually")

    def predict(self, user_idx: int, item_idx: int) -> float:       # Predict rating using weighted combination of model predictions
        """
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Weighted average prediction
        """
        if not self.is_fitted:
            raise ValueError("Hybrid model not fitted. Call fit() first.")
        
        predictions = []
        active_weights = []
        
        for model_name in self.model_names:
            try:
                pred = self.models[model_name].predict(user_idx, item_idx)
                predictions.append(pred)
                active_weights.append(self.weights[model_name])
            except Exception as e:
                # Skip models that can't make predictions
                print(f"Model {model_name} failed to predict: {e}")
                continue
        
        if not predictions:
            return 3.0  # Default rating
        
        # Normalize weights for active models
        total_active_weight = sum(active_weights)
        if total_active_weight == 0:
            return np.mean(predictions)
        
        normalized_weights = [w / total_active_weight for w in active_weights]
        weighted_prediction = sum(p * w for p, w in zip(predictions, normalized_weights))
        
        return np.clip(weighted_prediction, 1, 5)

    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[Tuple[int, float]]: # Recommend items using weighted hybrid approach
        """
        Recommend items using weighted hybrid approach
        """
        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def optimize_weights(self, validation_data: pd.DataFrame, 
                        user_item_matrix: np.ndarray) -> Dict[str, float]:  # Optimize weights based on validation performance
        
        from scipy.optimize import minimize
        
        def objective(weights):
            # Update weights
            weight_dict = {name: weights[i] for i, name in enumerate(self.model_names)}
            total_weight = sum(weight_dict.values())
            self.weights = {name: w / total_weight for name, w in weight_dict.items()}
            
            # Calculate validation error
            errors = []
            for _, row in validation_data.iterrows():
                try:
                    pred = self.predict(int(row['user_idx']), int(row['item_idx']))
                    error = (row['rating'] - pred) ** 2
                    errors.append(error)
                except:
                    continue
            
            return np.mean(errors) if errors else float('inf')
        
        # Initial weights
        initial_weights = [self.weights[name] for name in self.model_names]
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in self.model_names]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimized_weights = {name: result.x[i] for i, name in enumerate(self.model_names)}
            self.weights = optimized_weights
            print(f"Optimized weights: {optimized_weights}")
            return optimized_weights
        else:
            print("Weight optimization failed, keeping original weights")
            return self.weights
