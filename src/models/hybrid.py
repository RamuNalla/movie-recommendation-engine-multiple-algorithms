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
