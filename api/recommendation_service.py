import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class RecommendationService:
    """
    Service class that handles recommendation business logic
    """
    
    def __init__(self, models_dict: Dict[str, Any], user_item_matrix: np.ndarray,
                 movies_df: pd.DataFrame, preprocessor: Any):
        self.models_dict = models_dict
        self.user_item_matrix = user_item_matrix
        self.movies_df = movies_df
        self.preprocessor = preprocessor
        
        # Cache for frequently accessed data
        self._user_profiles_cache = {}
        self._item_popularity_cache = {}

    def get_recommendations(self, user_id: int, model_name: str = "WeightedHybrid", 
                          n_recommendations: int = 10, 
                          filter_seen: bool = True,
                          min_rating_threshold: float = 3.5) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User ID
            model_name: Name of the model to use
            n_recommendations: Number of recommendations
            filter_seen: Whether to filter out already seen items
            min_rating_threshold: Minimum predicted rating threshold
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Convert user ID to index
            user_idx = self.preprocessor.user_encoder.transform([user_id])[0]
            
            # Get model
            if model_name not in self.models_dict:
                raise ValueError(f"Model '{model_name}' not available")
            
            model = self.models_dict[model_name]
            
            # Generate recommendations
            raw_recommendations = model.recommend_items(
                user_idx, self.user_item_matrix, n_recommendations * 2  # Get more to filter
            )
            
            # Process and filter recommendations
            recommendations = []
            for item_idx, predicted_rating in raw_recommendations:
                # Skip if below threshold
                if predicted_rating < min_rating_threshold:
                    continue
                
                # Convert to original item ID
                original_item_id = self.preprocessor.item_encoder.classes_[item_idx]
                
                # Get movie details
                movie_info = self.movies_df[self.movies_df['item_id'] == original_item_id].iloc[0]
                
                # Calculate additional metrics
                confidence = self._calculate_prediction_confidence(user_idx, item_idx)
                popularity_score = self._get_item_popularity(item_idx)
                
                recommendation = {
                    "item_id": int(original_item_id),
                    "title": movie_info['title'],
                    "predicted_rating": float(predicted_rating),
                    "confidence": float(confidence),
                    "popularity_score": float(popularity_score),
                    "genres": self._extract_genres(movie_info),
                    "year": int(movie_info['year']) if pd.notna(movie_info['year']) else None,
                    "reason": self._generate_recommendation_reason(
                        user_idx, item_idx, model_name, predicted_rating
                    )
                }
                
                recommendations.append(recommendation)
                
                if len(recommendations) >= n_recommendations:
                    break
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            raise

    
    def predict_rating(self, user_id: int, item_id: int, 
                      model_name: str = "WeightedHybrid") -> Dict[str, Any]:
        """
        Predict rating for a specific user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            model_name: Model to use for prediction
            
        Returns:
            Prediction results dictionary
        """
        try:
            # Convert IDs to indices
            user_idx = self.preprocessor.user_encoder.transform([user_id])[0]
            item_idx = self.preprocessor.item_encoder.transform([item_id])[0]
            
            # Get prediction
            model = self.models_dict[model_name]
            predicted_rating = model.predict(user_idx, item_idx)
            
            # Calculate confidence and additional metrics
            confidence = self._calculate_prediction_confidence(user_idx, item_idx)
            user_profile = self._get_user_profile(user_idx)
            item_popularity = self._get_item_popularity(item_idx)
            
            return {
                "predicted_rating": float(predicted_rating),
                "confidence": float(confidence),
                "user_profile_size": user_profile["profile_size"],
                "item_popularity": float(item_popularity),
                "prediction_quality": "High" if confidence > 0.7 else 
                                    "Medium" if confidence > 0.4 else "Low"
            }
            
        except Exception as e:
            logger.error(f"Error predicting rating for user {user_id}, item {item_id}: {e}")
            raise

    def get_similar_items(self, item_id: int, n_similar: int = 10, 
                         similarity_type: str = "content") -> List[Dict[str, Any]]:
        """
        Get items similar to a given item
        
        Args:
            item_id: Target item ID
            n_similar: Number of similar items to return
            similarity_type: Type of similarity ("content" or "collaborative")
            
        Returns:
            List of similar items
        """
        try:
            # Convert to index
            item_idx = self.preprocessor.item_encoder.transform([item_id])[0]
            
            if similarity_type == "content" and "ContentBased" in self.models_dict:
                content_model = self.models_dict["ContentBased"]
                similar_items = content_model.get_similar_items(item_idx, n_similar)
                
            elif similarity_type == "collaborative" and "UserCF_Cosine" in self.models_dict:
                cf_model = self.models_dict["UserCF_Cosine"]
                similar_items = cf_model.get_item_neighbors(item_idx, n_similar)
                
            else:
                raise ValueError(f"Similarity type '{similarity_type}' not available")
            
            # Process results
            results = []
            for sim_item_idx, similarity_score in similar_items:
                original_item_id = self.preprocessor.item_encoder.classes_[sim_item_idx]
                movie_info = self.movies_df[self.movies_df['item_id'] == original_item_id].iloc[0]
                
                results.append({
                    "item_id": int(original_item_id),
                    "title": movie_info['title'],
                    "similarity_score": float(similarity_score),
                    "genres": self._extract_genres(movie_info),
                    "year": int(movie_info['year']) if pd.notna(movie_info['year']) else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar items for item {item_id}: {e}")
            raise

    
    def get_trending_items(self, time_window_days: int = 30, 
                          n_items: int = 20) -> List[Dict[str, Any]]:
        """
        Get trending items (simplified implementation)
        In production, this would use real-time interaction data
        
        Args:
            time_window_days: Time window for trend calculation
            n_items: Number of trending items to return
            
        Returns:
            List of trending items
        """
        # For demo purposes, return popular items with some randomization
        item_popularities = np.sum(self.user_item_matrix > 0, axis=0)
        
        # Add some randomness to simulate trending
        np.random.seed(int(datetime.now().timestamp()) // (24 * 3600))  # Daily seed
        trending_scores = item_popularities + np.random.normal(0, 10, len(item_popularities))
        
        # Get top trending items
        top_indices = np.argsort(trending_scores)[-n_items:][::-1]
        
        trending_items = []
        for item_idx in top_indices:
            original_item_id = self.preprocessor.item_encoder.classes_[item_idx]
            movie_info = self.movies_df[self.movies_df['item_id'] == original_item_id].iloc[0]
            
            trending_items.append({
                "item_id": int(original_item_id),
                "title": movie_info['title'],
                "trending_score": float(trending_scores[item_idx]),
                "popularity": int(item_popularities[item_idx]),
                "genres": self._extract_genres(movie_info),
                "year": int(movie_info['year']) if pd.notna(movie_info['year']) else None
            })
        
        return trending_items
    
    def get_user_profile_analysis(self, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive user profile analysis
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user profile analysis
        """
        try:
            user_idx = self.preprocessor.user_encoder.transform([user_id])[0]
            user_ratings = self.user_item_matrix[user_idx]
            rated_items = np.where(user_ratings > 0)[0]
            
            if len(rated_items) == 0:
                return {
                    "user_id": user_id,
                    "total_ratings": 0,
                    "message": "User has no ratings"
                }
            
            # Basic statistics
            ratings = user_ratings[rated_items]
            
            # Genre preferences
            genre_preferences = self._calculate_genre_preferences(user_idx, rated_items)
            
            # Rating patterns
            rating_distribution = {
                int(rating): int(np.sum(ratings == rating)) 
                for rating in np.unique(ratings)
            }
            
            return {
                "user_id": user_id,
                "total_ratings": int(len(rated_items)),
                "average_rating": float(np.mean(ratings)),
                "rating_std": float(np.std(ratings)),
                "rating_distribution": rating_distribution,
                "genre_preferences": genre_preferences,
                "activity_level": self._categorize_user_activity(len(rated_items)),
                "rating_bias": self._calculate_rating_bias(ratings)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user profile {user_id}: {e}")
            raise

    def _calculate_prediction_confidence(self, user_idx: int, item_idx: int) -> float:
        """Calculate confidence score for a prediction"""
        # User activity level
        user_activity = np.sum(self.user_item_matrix[user_idx] > 0)
        user_confidence = min(user_activity / 50.0, 1.0)
        
        # Item popularity
        item_popularity = np.sum(self.user_item_matrix[:, item_idx] > 0)
        item_confidence = min(item_popularity / 50.0, 1.0)
        
        # Combined confidence
        overall_confidence = (user_confidence + item_confidence) / 2
        return overall_confidence
    
    def _get_user_profile(self, user_idx: int) -> Dict[str, Any]:
        """Get user profile information with caching"""
        if user_idx in self._user_profiles_cache:
            return self._user_profiles_cache[user_idx]
        
        user_ratings = self.user_item_matrix[user_idx]
        rated_items = user_ratings[user_ratings > 0]
        
        profile = {
            "profile_size": int(len(rated_items)),
            "average_rating": float(np.mean(rated_items)) if len(rated_items) > 0 else 0.0,
            "rating_std": float(np.std(rated_items)) if len(rated_items) > 1 else 0.0
        }
        
        self._user_profiles_cache[user_idx] = profile
        return profile
    
    def _get_item_popularity(self, item_idx: int) -> float:
        """Get item popularity score with caching"""
        if item_idx in self._item_popularity_cache:
            return self._item_popularity_cache[item_idx]
        
        popularity = np.sum(self.user_item_matrix[:, item_idx] > 0)
        self._item_popularity_cache[item_idx] = popularity
        return popularity
    
    def _extract_genres(self, movie_info: pd.Series) -> List[str]:
        """Extract genres from movie information"""
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        return [genre for genre in genre_cols if movie_info[genre] == 1]
    
    def _generate_recommendation_reason(self, user_idx: int, item_idx: int, 
                                      model_name: str, predicted_rating: float) -> str:
        """Generate explanation for why an item was recommended"""
        reasons = []
        
        if "UserCF" in model_name:
            reasons.append("Users with similar taste also liked this movie")
        elif "ContentBased" in model_name:
            reasons.append("Based on movies you've enjoyed with similar content")
        elif "SVD" in model_name:
            reasons.append("Discovered through pattern analysis of your preferences")
        elif "Hybrid" in model_name:
            reasons.append("Recommended by our advanced hybrid algorithm")
        else:
            reasons.append("Recommended based on your viewing history")
        
        if predicted_rating >= 4.5:
            reasons.append("Predicted to be a great match for you")
        elif predicted_rating >= 4.0:
            reasons.append("Likely to match your preferences")
        
        return ". ".join(reasons) if reasons else "Recommended for you"
    
    def _calculate_genre_preferences(self, user_idx: int, rated_items: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate user's genre preferences"""
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genre_ratings = defaultdict(list)
        
        for item_idx in rated_items:
            original_item_id = self.preprocessor.item_encoder.classes_[item_idx]
            movie_info = self.movies_df[self.movies_df['item_id'] == original_item_id].iloc[0]
            rating = self.user_item_matrix[user_idx, item_idx]
            
            for genre in genre_cols:
                if movie_info[genre] == 1:
                    genre_ratings[genre].append(rating)
        
        # Calculate average ratings per genre
        genre_preferences = []
        for genre, ratings in genre_ratings.items():
            if len(ratings) > 0:
                genre_preferences.append({
                    "genre": genre,
                    "average_rating": float(np.mean(ratings)),
                    "count": len(ratings)
                })
        
        # Sort by average rating
        genre_preferences.sort(key=lambda x: x["average_rating"], reverse=True)
        
        return genre_preferences[:5]  # Return top 5 genres
    

    def _categorize_user_activity(self, num_ratings: int) -> str:
        """Categorize user activity level"""
        if num_ratings >= 100:
            return "Very Active"
        elif num_ratings >= 50:
            return "Active"
        elif num_ratings >= 20:
            return "Moderate"
        elif num_ratings >= 5:
            return "Light"
        else:
            return "New User"
        
    def _calculate_rating_bias(self, ratings: np.ndarray) -> str:
        """Calculate if user tends to rate high or low"""
        mean_rating = np.mean(ratings)
        
        if mean_rating >= 4.0:
            return "Positive (tends to rate high)"
        elif mean_rating >= 3.5:
            return "Slightly Positive"
        elif mean_rating >= 3.0:
            return "Neutral"
        elif mean_rating >= 2.5:
            return "Slightly Negative"
        else:
            return "Negative (tends to rate low)"
        
    
    def compare_models(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """
        Compare predictions from all available models
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Dictionary with model comparisons
        """
        try:
            user_idx = self.preprocessor.user_encoder.transform([user_id])[0]
            item_idx = self.preprocessor.item_encoder.transform([item_id])[0]
            
            predictions = {}
            valid_predictions = []
            
            for model_name, model in self.models_dict.items():
                try:
                    pred = model.predict(user_idx, item_idx)
                    predictions[model_name] = float(pred)
                    valid_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Model {model_name} failed to predict: {e}")
                    predictions[model_name] = None
            
            # Calculate statistics
            if valid_predictions:
                ensemble_prediction = float(np.mean(valid_predictions))
                prediction_variance = float(np.var(valid_predictions))
                min_prediction = float(np.min(valid_predictions))
                max_prediction = float(np.max(valid_predictions))
            else:
                ensemble_prediction = 3.0
                prediction_variance = 0.0
                min_prediction = 3.0
                max_prediction = 3.0
            
            return {
                "user_id": user_id,
                "item_id": item_id,
                "predictions": predictions,
                "ensemble_prediction": ensemble_prediction,
                "prediction_variance": prediction_variance,
                "prediction_range": {
                    "min": min_prediction,
                    "max": max_prediction,
                    "spread": max_prediction - min_prediction
                },
                "agreement_level": "High" if prediction_variance < 0.5 else 
                                 "Medium" if prediction_variance < 1.0 else "Low"
            }
            
        except Exception as e:
            logger.error(f"Error comparing models for user {user_id}, item {item_id}: {e}")
            raise