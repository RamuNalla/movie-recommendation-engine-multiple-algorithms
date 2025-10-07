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
        result = minimize(objective, initial_weights, method='SLSQP',   # Sequential least squares Programming algorithm
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimized_weights = {name: result.x[i] for i, name in enumerate(self.model_names)}
            self.weights = optimized_weights
            print(f"Optimized weights: {optimized_weights}")
            return optimized_weights
        else:
            print("Weight optimization failed, keeping original weights")
            return self.weights
        
class SwitchingHybrid:  # Switching Hybrid Recommender System. Switches between different models based on user/item characteristics

    def __init__(self, models: Dict[str, Any], switching_criteria: str = 'user_profile_size'):
        """
        Initialize switching hybrid model
        
        Args:
            models: Dictionary of {model_name: model_instance}
            switching_criteria: Criteria for switching ('user_profile_size', 'item_popularity', etc.)
        """
        self.models = models
        self.switching_criteria = switching_criteria
        self.model_names = list(models.keys())
        self.user_profile_sizes = None
        self.item_popularity = None

    def fit(self, user_item_matrix: np.ndarray, ratings_df: pd.DataFrame):
        """
        Fit switching criteria and prepare for model selection
        
        Args:
            user_item_matrix: User-item rating matrix
            ratings_df: Ratings DataFrame for calculating statistics
        """
        
        self.user_profile_sizes = np.sum(user_item_matrix > 0, axis=1)      # # Calculate user profile sizes (number of ratings per user)
        
        item_counts = ratings_df['item_idx'].value_counts().to_dict()       # Calculate item popularity (number of ratings per item)
        self.item_popularity = {i: item_counts.get(i, 0) 
                              for i in range(user_item_matrix.shape[1])}
        
        print(f"Switching Hybrid fitted with {len(self.models)} models")

    def _select_model(self, user_idx: int, item_idx: int) -> str:           # Select appropriate model based on switching criteria
        """
        Select appropriate model based on switching criteria
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Selected model name
        """
        if self.switching_criteria == 'user_profile_size':
            # Use collaborative filtering for users with many ratings,
            # content-based for users with few ratings
            user_ratings_count = self.user_profile_sizes[user_idx]
            
            if user_ratings_count >= 20:        # Heavy user
                return 'collaborative_filtering' if 'collaborative_filtering' in self.models else self.model_names[0]
            else:                               # Light user
                return 'content_based' if 'content_based' in self.models else self.model_names[-1]
        
        elif self.switching_criteria == 'item_popularity':
            # Use collaborative filtering for popular items,
            # content-based for unpopular items
            item_rating_count = self.item_popularity.get(item_idx, 0)
            
            if item_rating_count >= 50:     # Popular item
                return 'collaborative_filtering' if 'collaborative_filtering' in self.models else self.model_names[0]
            else:                           # Unpopular item
                return 'content_based' if 'content_based' in self.models else self.model_names[-1]
        
        else:
            return self.model_names[0]      # Default to first model
        

    def predict(self, user_idx: int, item_idx: int) -> float:       # Predict rating using selected model
        """
        Args:
            user_idx: User index
            item_idx: Item index
        Returns:
            Prediction from selected model
        """
        selected_model = self._select_model(user_idx, item_idx)
        
        try:
            return self.models[selected_model].predict(user_idx, item_idx)
        except Exception as e:
            print(f"Selected model {selected_model} failed: {e}")
            for model_name, model in self.models.items():       # Fallback to first available model
                try:
                    return model.predict(user_idx, item_idx)
                except:
                    continue
            return 3.0  # Ultimate fallback
        
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:     # Recommend items using switching hybrid approach
        
        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    

class MixedHybrid:      # Presents recommendations from different models simultaneously
    # Mixed hybrid recommendation system runs different recommendation algorithms and presents the results together
    def __init__(self, models: Dict[str, Any], mix_ratios: Optional[Dict[str, float]] = None):
        """
        Initialize mixed hybrid model
        
        Args:
            models: Dictionary of {model_name: model_instance}
            mix_ratios: Dictionary of {model_name: ratio} for mixing recommendations
        """
        self.models = models
        self.model_names = list(models.keys())
        
        if mix_ratios is None:      # Equal mixing ratios
            self.mix_ratios = {name: 1.0 / len(models) for name in self.model_names}
        else:                      # Normalize mixing ratios
            total_ratio = sum(mix_ratios.values())
            self.mix_ratios = {name: ratio / total_ratio for name, ratio in mix_ratios.items()}

    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict using first available model (mixed hybrid focuses on recommendations)
        """
        for model in self.models.values():
            try:
                return model.predict(user_idx, item_idx)
            except:
                continue
        return 3.0

    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Mix recommendations from different models
        
        Args:
            user_idx: User index
            user_item_matrix: User-item matrix
            n_recommendations: Total number of recommendations
            
        Returns:
            Mixed list of recommendations
        """
        all_recommendations = {}
        
        # Get recommendations from each model
        for model_name, model in self.models.items():
            try:
                model_recs = model.recommend_items(user_idx, user_item_matrix, n_recommendations * 2)
                # Store with model source info
                for item_idx, score in model_recs:
                    if item_idx not in all_recommendations:
                        all_recommendations[item_idx] = []
                    all_recommendations[item_idx].append((model_name, score))
            except Exception as e:
                print(f"Model {model_name} failed to generate recommendations: {e}")
                continue
        
        # Calculate number of recommendations from each model
        model_rec_counts = {}
        for model_name in self.model_names:
            model_rec_counts[model_name] = int(n_recommendations * self.mix_ratios[model_name])
        
        # Ensure total adds up to n_recommendations
        remaining = n_recommendations - sum(model_rec_counts.values())
        if remaining > 0:
            # Add remaining to first model
            model_rec_counts[self.model_names[0]] += remaining
        
        # Select recommendations maintaining diversity
        final_recommendations = []
        used_items = set()
        
        # Round-robin selection from each model
        model_recommendation_lists = {}
        for model_name in self.model_names:
            model_items = []
            for item_idx, model_scores in all_recommendations.items():
                for model, score in model_scores:
                    if model == model_name:
                        model_items.append((item_idx, score))
                        break
            model_items.sort(key=lambda x: x[1], reverse=True)
            model_recommendation_lists[model_name] = model_items
        
        # Fill recommendations according to ratios
        for model_name, count in model_rec_counts.items():
            added_count = 0
            for item_idx, score in model_recommendation_lists.get(model_name, []):
                if item_idx not in used_items and added_count < count:
                    final_recommendations.append((item_idx, score))
                    used_items.add(item_idx)
                    added_count += 1
                    if len(final_recommendations) >= n_recommendations:
                        break
            if len(final_recommendations) >= n_recommendations:
                break
        
        return final_recommendations[:n_recommendations]
    

class FeatureCombinationHybrid:     # Feature Combination Hybrid using Machine Learning. Combines features from different recommendation approaches

    def __init__(self, base_models: Dict[str, Any], meta_learner='random_forest'):
        """
        Initialize feature combination hybrid
        
        Args:
            base_models: Dictionary of base recommendation models
            meta_learner: Type of meta-learner ('random_forest', 'linear_regression')
        """
        self.base_models = base_models
        self.model_names = list(base_models.keys())
        
        if meta_learner == 'random_forest':
            self.meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
        elif meta_learner == 'linear_regression':
            self.meta_learner = LinearRegression()
        else:
            raise ValueError(f"Unsupported meta-learner: {meta_learner}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False

    
    def _extract_features(self, user_idx: int, item_idx: int, 
                         user_item_matrix: np.ndarray) -> np.ndarray:
        """
        Extract features from base models and user-item interactions
        
        Args:
            user_idx: User index
            item_idx: Item index
            user_item_matrix: User-item matrix
            
        Returns:
            Feature vector
        """
        features = []
        
        # Predictions from base models
        for model_name, model in self.base_models.items():
            try:
                pred = model.predict(user_idx, item_idx)
                features.append(pred)
            except:
                features.append(3.0)  # Default value
        
        # User features
        user_ratings = user_item_matrix[user_idx]
        user_mean_rating = np.mean(user_ratings[user_ratings > 0]) if np.sum(user_ratings > 0) > 0 else 3.0
        user_num_ratings = np.sum(user_ratings > 0)
        user_rating_std = np.std(user_ratings[user_ratings > 0]) if np.sum(user_ratings > 0) > 1 else 0.0
        
        features.extend([user_mean_rating, user_num_ratings, user_rating_std])
        
        # Item features
        item_ratings = user_item_matrix[:, item_idx]
        item_mean_rating = np.mean(item_ratings[item_ratings > 0]) if np.sum(item_ratings > 0) > 0 else 3.0
        item_num_ratings = np.sum(item_ratings > 0)
        item_rating_std = np.std(item_ratings[item_ratings > 0]) if np.sum(item_ratings > 0) > 1 else 0.0
        
        features.extend([item_mean_rating, item_num_ratings, item_rating_std])
        
        return np.array(features)
    

    def fit(self, train_data: pd.DataFrame, user_item_matrix: np.ndarray):
        """
        Fit the feature combination hybrid model
        
        Args:
            train_data: Training data with user_idx, item_idx, rating columns
            user_item_matrix: User-item rating matrix
        """
        # Extract features for training data
        X_train = []
        y_train = []
        
        print("Extracting features for meta-learner training...")
        for i, (_, row) in enumerate(train_data.iterrows()):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(train_data)} samples")
            
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            rating = row['rating']
            
            try:
                features = self._extract_features(user_idx, item_idx, user_item_matrix)
                X_train.append(features)
                y_train.append(rating)
            except Exception as e:
                continue
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta_learner.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        print("Feature combination hybrid model fitted successfully!")

    
    def predict(self, user_idx: int, item_idx: int, user_item_matrix: np.ndarray) -> float:     # Predict rating using feature combination approach
        """       
        Args:
            user_idx: User index
            item_idx: Item index
            user_item_matrix: User-item matrix
            
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract features
        features = self._extract_features(user_idx, item_idx, user_item_matrix)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict using meta-learner
        prediction = self.meta_learner.predict(features_scaled)[0]
        
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:     # Recommend items using feature combination approach
        
        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx, user_item_matrix)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_feature_importance(self) -> Dict[str, float]:       # Get feature importance from the meta-learner
        """
        Returns:
            Dictionary with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.meta_learner, 'feature_importances_'):
            importances = self.meta_learner.feature_importances_
            feature_names = (self.model_names + 
                           ['user_mean_rating', 'user_num_ratings', 'user_rating_std',
                            'item_mean_rating', 'item_num_ratings', 'item_rating_std'])
            
            return dict(zip(feature_names, importances))
        else:
            return {}

class CascadeHybrid:            # Cascade Hybrid Recommender System. Uses models in sequence, with later models refining earlier predictions

    def __init__(self, models: List[Tuple[str, Any]], refinement_threshold: float = 0.5):
        """
        Initialize cascade hybrid model
        
        Args:
            models: List of (model_name, model_instance) tuples in order of cascade
            refinement_threshold: Confidence threshold for moving to next model
        """
        self.models = models
        self.model_names = [name for name, _ in models]
        self.refinement_threshold = refinement_threshold

    def predict(self, user_idx: int, item_idx: int, user_item_matrix: np.ndarray = None) -> float:
        """
        Predict using cascade of models
        
        Args:
            user_idx: User index
            item_idx: Item index
            user_item_matrix: User-item matrix (used for confidence calculation)
            
        Returns:
            Prediction from cascade
        """
        for i, (model_name, model) in enumerate(self.models):
            try:
                prediction = model.predict(user_idx, item_idx)
                
                # Calculate confidence (simplified approach)
                confidence = self._calculate_confidence(user_idx, item_idx, model_name, user_item_matrix)
                
                # If confident enough or last model, return prediction
                if confidence >= self.refinement_threshold or i == len(self.models) - 1:
                    return prediction
                
            except Exception as e:
                print(f"Model {model_name} failed in cascade: {e}")
                continue
        
        return 3.0  # Fallback

    def _calculate_confidence(self, user_idx: int, item_idx: int, model_name: str, 
                            user_item_matrix: np.ndarray) -> float:
        """
        Calculate prediction confidence (simplified heuristic)
        
        Args:
            user_idx: User index
            item_idx: Item index
            model_name: Name of the model
            user_item_matrix: User-item matrix
            
        Returns:
            Confidence score between 0 and 1
        """
        if user_item_matrix is None:
            return 1.0  # Always confident if no matrix provided
        
        # Simple heuristics for confidence
        if 'collaborative' in model_name.lower():
            # CF confidence based on user profile size and item popularity
            user_ratings = np.sum(user_item_matrix[user_idx] > 0)
            item_ratings = np.sum(user_item_matrix[:, item_idx] > 0)
            confidence = min(user_ratings / 50.0, 1.0) * min(item_ratings / 50.0, 1.0)
        
        elif 'content' in model_name.lower():
            # Content-based always confident if item exists
            confidence = 0.8
        
        else:
            # Default confidence
            confidence = 0.6
        
        return confidence