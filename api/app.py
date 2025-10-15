# api/app.py
"""
FastAPI Web Service for Movie Recommendation System
Provides RESTful endpoints for getting recommendations
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append('../src')

from data.data_loader import MovielensDataloader
from models.hybrid import WeightedHybrid

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="A comprehensive movie recommendation system using various ML algorithms",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
models_dict = {}
user_item_matrix = None
movies_df = None
preprocessor = None
model_info = {}
ratings = None

# Pydantic models for API
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10
    model_name: str = "WeightedHybrid"

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    model_used: str
    metadata: Dict[str, Any]

    class Config:
        # This allows arbitrary types and disables validation for the response
        arbitrary_types_allowed = True

class UserRatingRequest(BaseModel):
    user_id: int
    item_id: int
    rating: float

class PredictionResponse(BaseModel):
    user_id: int
    item_id: int
    predicted_rating: float
    model_used: str
    confidence: Optional[float] = None

class ModelComparisonResponse(BaseModel):
    user_id: int
    item_id: int
    predictions: Dict[str, float]
    ensemble_prediction: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    dataset_info: Dict[str, Any]

# Load models and data on startup
@app.on_event("startup")
async def load_models_and_data():
    """Load trained models and data on application startup"""
    global models_dict, user_item_matrix, movies_df, preprocessor, model_info, ratings
    
    try:
        # Load data
        print("Loading data...")
        loader = MovielensDataloader('../data/raw/ml-100k/')
        ratings, movies, users = loader.load_all_data()
        movies_df = movies
        
        # Load preprocessor and matrices (would be saved during training)
        # For demo purposes, we'll recreate them
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        user_item_matrix, matrix_info = preprocessor.create_user_item_matrix(ratings)
        
        # Load or create models (in production, these would be loaded from disk)
        print("Loading models...")
        
        # For demo, create a simple model
        from models.collaborative_filtering import CollaborativeFiltering
        from models.matrix_factorization import SVDRecommender
        from models.content_based import ContentBasedRecommender
        
        # Train simple models (in production, load from saved files)
        train_data, _ = loader.get_train_test_split()
        train_ratings = train_data.copy()
        train_ratings['user_idx'] = preprocessor.user_encoder.transform(train_ratings['user_id'])
        train_ratings['item_idx'] = preprocessor.item_encoder.transform(train_ratings['item_id'])
        
        train_matrix = np.zeros_like(user_item_matrix)
        for _, row in train_ratings.iterrows():
            train_matrix[int(row['user_idx']), int(row['item_idx'])] = row['rating']
        
        # Initialize models
        cf_model = CollaborativeFiltering(similarity_metric='cosine', n_neighbors=50)
        cf_model.fit(train_matrix)
        
        svd_model = SVDRecommender(n_factors=50, random_state=42)
        svd_model.fit(train_matrix)
        
        content_model = ContentBasedRecommender()
        content_model.fit(movies, train_ratings)
        
        # Create hybrid model
        base_models = {
            'UserCF': cf_model,
            'SVD': svd_model,
            'ContentBased': content_model
        }
        
        hybrid_model = WeightedHybrid(
            models=base_models,
            weights={'UserCF': 0.4, 'SVD': 0.4, 'ContentBased': 0.2}
        )
        hybrid_model.fit()
        
        models_dict = {
            'UserCF': cf_model,
            'SVD': svd_model,
            'ContentBased': content_model,
            'WeightedHybrid': hybrid_model
        }
        
        model_info = {
            'n_users': matrix_info['n_users'],
            'n_items': matrix_info['n_items'],
            'n_ratings': matrix_info['n_ratings'],
            'sparsity': matrix_info['sparsity']
        }
        
        print("Models and data loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models and data: {e}")
        raise e

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models_dict.keys()),
        dataset_info=model_info
    )

@app.get("/models")
async def get_available_models():
    """Get list of available recommendation models"""
    return {
        "available_models": list(models_dict.keys()),
        "default_model": "WeightedHybrid",
        "model_descriptions": {
            "UserCF": "User-based Collaborative Filtering",
            "SVD": "Singular Value Decomposition Matrix Factorization",
            "ContentBased": "Content-based filtering using movie features",
            "WeightedHybrid": "Weighted combination of multiple models"
        }
    }

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user"""
    try:
        # Basic validation
        if request.user_id < 1:
            raise HTTPException(status_code=400, detail="User ID must be positive")
        
        if request.model_name not in models_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' not available. Available: {list(models_dict.keys())}"
            )
        
        # Convert user ID to index with proper error handling
        try:
            if request.user_id not in preprocessor.user_encoder.classes_:
                valid_range = f"1 to {max(preprocessor.user_encoder.classes_)}"
                raise HTTPException(
                    status_code=404,
                    detail=f"User ID {request.user_id} not found. Valid range: {valid_range}"
                )
            
            user_idx = preprocessor.user_encoder.transform([request.user_id])[0]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting user ID: {str(e)}"
            )
        
        # Get recommendations
        try:
            model = models_dict[request.model_name]
            raw_recommendations = model.recommend_items(
                user_idx, user_item_matrix, request.n_recommendations
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating recommendations: {str(e)}"
            )
        
        # Process recommendations
        recommendations = []
        for item_idx, score in raw_recommendations:
            try:
                original_item_id = preprocessor.item_encoder.classes_[item_idx]
                movie_info = movies_df[movies_df['item_id'] == original_item_id].iloc[0]
                
                genres = [genre for genre in ['Action', 'Adventure', 'Animation', 'Children',
                          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                          'Sci-Fi', 'Thriller', 'War', 'Western'] 
                          if movie_info[genre] == 1]
                
                recommendations.append({
                    "item_id": int(original_item_id),
                    "title": str(movie_info['title']),
                    "predicted_rating": float(score),
                    "genres": genres,
                    "year": int(movie_info['year']) if pd.notna(movie_info['year']) else None,
                    "imdb_url": str(movie_info['imdb_url']) if pd.notna(movie_info['imdb_url']) else None
                })
            except Exception as e:
                print(f"Warning: Could not process item {item_idx}: {e}")
                continue
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations could be generated for this user"
            )
        
        return {
            "user_id": int(request.user_id),
            "recommendations": recommendations,
            "model_used": str(request.model_name),
            "metadata": {
                "total_recommendations": len(recommendations),
                "user_profile_size": int(np.sum(user_item_matrix[user_idx] > 0)),
                "average_predicted_rating": float(np.mean([r["predicted_rating"] for r in recommendations]))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("=" * 60)
        print("ERROR in get_recommendations:")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/predict")
async def predict_rating(request: UserRatingRequest):
    """Predict rating for a specific user-item pair"""
    try:
        # Validate inputs
        if request.user_id < 1:
            raise HTTPException(status_code=400, detail="User ID must be positive")
        
        if request.item_id < 1:
            raise HTTPException(status_code=400, detail="Item ID must be positive")
        
        # Convert IDs to indices
        try:
            if request.user_id not in preprocessor.user_encoder.classes_:
                raise HTTPException(status_code=404, detail=f"User ID {request.user_id} not found")
            
            if request.item_id not in preprocessor.item_encoder.classes_:
                raise HTTPException(status_code=404, detail=f"Item ID {request.item_id} not found")
            
            user_idx = preprocessor.user_encoder.transform([request.user_id])[0]
            item_idx = preprocessor.item_encoder.transform([request.item_id])[0]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting IDs: {str(e)}")
        
        # Use hybrid model for prediction
        model = models_dict['WeightedHybrid']
        predicted_rating = model.predict(user_idx, item_idx)
        
        # Calculate confidence
        user_activity = np.sum(user_item_matrix[user_idx] > 0)
        item_popularity = np.sum(user_item_matrix[:, item_idx] > 0)
        confidence = min((user_activity / 50.0) * (item_popularity / 50.0), 1.0)
        
        return {
            "user_id": int(request.user_id),
            "item_id": int(request.item_id),
            "predicted_rating": float(predicted_rating),
            "model_used": "WeightedHybrid",
            "confidence": float(confidence)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/compare-models")
async def compare_model_predictions(request: UserRatingRequest):
    """Compare predictions from all available models"""
    try:
        # Convert IDs to indices
        try:
            if request.user_id not in preprocessor.user_encoder.classes_:
                valid_range = f"1 to {max(preprocessor.user_encoder.classes_)}"
                raise HTTPException(
                    status_code=404,
                    detail=f"User ID {request.user_id} not found. Valid range: {valid_range}"
                )
            
            user_idx = preprocessor.user_encoder.transform([request.user_id])[0]
            item_idx = preprocessor.item_encoder.transform([request.item_id])[0]
            
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"User ID {request.user_id} or Item ID {request.item_id} not found"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting IDs: {str(e)}"
            )
        
        # Get predictions from all models
        predictions = {}
        valid_predictions = []
        
        for model_name, model in models_dict.items():
            try:
                pred = model.predict(user_idx, item_idx)
                predictions[model_name] = float(pred)
                valid_predictions.append(pred)
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                predictions[model_name] = None
        
        # Calculate ensemble prediction
        ensemble_prediction = float(np.mean(valid_predictions)) if valid_predictions else 3.0
        
        return {
            "user_id": int(request.user_id),
            "item_id": int(request.item_id),
            "predictions": predictions,
            "ensemble_prediction": ensemble_prediction
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("=" * 60)
        print("ERROR in compare_model_predictions:")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Get user profile information"""
    try:
        # Validate user ID
        if user_id < 1:
            raise HTTPException(status_code=400, detail="User ID must be positive")
        
        # Convert to index
        try:
            if user_id not in preprocessor.user_encoder.classes_:
                raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")
            
            user_idx = preprocessor.user_encoder.transform([user_id])[0]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting user ID: {str(e)}")
        
        # Get user ratings
        user_ratings = user_item_matrix[user_idx]
        rated_items = np.where(user_ratings > 0)[0]
        
        # Get rating statistics
        ratings_stats = {
            "total_ratings": int(len(rated_items)),
            "average_rating": float(np.mean(user_ratings[rated_items])) if len(rated_items) > 0 else 0.0,
            "rating_std": float(np.std(user_ratings[rated_items])) if len(rated_items) > 1 else 0.0,
            "min_rating": float(np.min(user_ratings[rated_items])) if len(rated_items) > 0 else 0.0,
            "max_rating": float(np.max(user_ratings[rated_items])) if len(rated_items) > 0 else 0.0
        }
        
        # Get favorite genres
        genre_preferences = {}
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        for genre in genre_cols:
            genre_ratings = []
            for item_idx in rated_items:
                original_item_id = preprocessor.item_encoder.classes_[item_idx]
                movie = movies_df[movies_df['item_id'] == original_item_id].iloc[0]
                if movie[genre] == 1:
                    genre_ratings.append(user_ratings[item_idx])
            
            if genre_ratings:
                genre_preferences[genre] = {
                    "average_rating": float(np.mean(genre_ratings)),
                    "count": len(genre_ratings)
                }
        
        # Sort genres by average rating
        favorite_genres = sorted(
            [(genre, info["average_rating"]) for genre, info in genre_preferences.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "user_id": user_id,
            "rating_statistics": ratings_stats,
            "favorite_genres": [{"genre": genre, "avg_rating": rating} for genre, rating in favorite_genres],
            "activity_level": "High" if ratings_stats["total_ratings"] > 100 else 
                           "Medium" if ratings_stats["total_ratings"] > 20 else "Low"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/movie/{item_id}/info")
async def get_movie_info(item_id: int):
    """Get detailed movie information"""
    try:
        # Find movie
        movie = movies_df[movies_df['item_id'] == item_id]
        if movie.empty:
            raise HTTPException(status_code=404, detail=f"Movie with ID {item_id} not found")
        
        movie_info = movie.iloc[0]
        
        # Get movie statistics
        try:
            if item_id in preprocessor.item_encoder.classes_:
                item_idx = preprocessor.item_encoder.transform([item_id])[0]
                item_ratings = user_item_matrix[:, item_idx]
                rated_users = np.where(item_ratings > 0)[0]
                
                rating_stats = {
                    "total_ratings": int(len(rated_users)),
                    "average_rating": float(np.mean(item_ratings[rated_users])) if len(rated_users) > 0 else 0.0,
                    "rating_std": float(np.std(item_ratings[rated_users])) if len(rated_users) > 1 else 0.0
                }
            else:
                rating_stats = {"total_ratings": 0, "average_rating": 0.0, "rating_std": 0.0}
        except Exception as e:
            print(f"Warning: Could not get rating stats: {e}")
            rating_stats = {"total_ratings": 0, "average_rating": 0.0, "rating_std": 0.0}
        
        # Extract genres
        genres = [genre for genre in ['Action', 'Adventure', 'Animation', 'Children',
                  'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                  'Sci-Fi', 'Thriller', 'War', 'Western'] 
                  if movie_info[genre] == 1]
        
        return {
            "item_id": int(item_id),
            "title": str(movie_info['title']),
            "genres": genres,
            "year": int(movie_info['year']) if pd.notna(movie_info['year']) else None,
            "release_date": str(movie_info['release_date']) if pd.notna(movie_info['release_date']) else None,
            "imdb_url": str(movie_info['imdb_url']) if pd.notna(movie_info['imdb_url']) else None,
            "rating_statistics": rating_stats,
            "popularity_rank": "High" if rating_stats["total_ratings"] > 100 else 
                             "Medium" if rating_stats["total_ratings"] > 20 else "Low"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/similar-movies/{item_id}")
async def get_similar_movies(item_id: int, n_similar: int = Query(default=5, ge=1, le=20)):
    """Get movies similar to a given movie"""
    try:
        # Check if content-based model is available
        if 'ContentBased' not in models_dict:
            raise HTTPException(
                status_code=503,
                detail="Content-based model not available for similarity computation"
            )
        
        # Convert to index
        try:
            item_idx = preprocessor.item_encoder.transform([item_id])[0]
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Movie with ID {item_id} not found"
            )
        
        # Get similar items from content-based model
        content_model = models_dict['ContentBased']
        similar_items = content_model.get_similar_items(item_idx, n_similar)
        
        # Convert to movie details
        similar_movies = []
        for sim_item_idx, similarity_score in similar_items:
            original_item_id = preprocessor.item_encoder.classes_[sim_item_idx]
            movie_info = movies_df[movies_df['item_id'] == original_item_id].iloc[0]
            
            similar_movies.append({
                "item_id": int(original_item_id),
                "title": movie_info['title'],
                "similarity_score": float(similarity_score),
                "genres": [genre for genre in ['Action', 'Adventure', 'Animation', 'Children',
                          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                          'Sci-Fi', 'Thriller', 'War', 'Western'] 
                          if movie_info[genre] == 1],
                "year": int(movie_info['year']) if pd.notna(movie_info['year']) else None
            })
        
        return {
            "target_movie_id": item_id,
            "similar_movies": similar_movies,
            "similarity_method": "content_based"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats/dataset")
async def get_dataset_statistics():
    """Get comprehensive dataset statistics"""
    try:
        # Basic statistics
        n_users, n_items = user_item_matrix.shape
        n_ratings = np.sum(user_item_matrix > 0)
        sparsity = 1 - (n_ratings / (n_users * n_items))
        
        # Rating distribution
        all_ratings = user_item_matrix[user_item_matrix > 0]
        rating_dist = {
            str(i): int(np.sum(all_ratings == i)) for i in range(1, 6)
        }
        
        # User activity statistics
        user_activities = np.sum(user_item_matrix > 0, axis=1)
        user_stats = {
            "mean_ratings_per_user": float(np.mean(user_activities)),
            "median_ratings_per_user": float(np.median(user_activities)),
            "std_ratings_per_user": float(np.std(user_activities)),
            "min_ratings_per_user": int(np.min(user_activities)),
            "max_ratings_per_user": int(np.max(user_activities))
        }
        
        # Item popularity statistics
        item_popularities = np.sum(user_item_matrix > 0, axis=0)
        item_stats = {
            "mean_ratings_per_item": float(np.mean(item_popularities)),
            "median_ratings_per_item": float(np.median(item_popularities)),
            "std_ratings_per_item": float(np.std(item_popularities)),
            "min_ratings_per_item": int(np.min(item_popularities)),
            "max_ratings_per_item": int(np.max(item_popularities))
        }
        
        # Genre statistics
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        genre_counts = {genre: int(movies_df[genre].sum()) for genre in genre_cols}
        
        return {
            "basic_statistics": {
                "n_users": n_users,
                "n_items": n_items,
                "n_ratings": int(n_ratings),
                "sparsity": float(sparsity),
                "density": float(1 - sparsity)
            },
            "rating_distribution": rating_dist,
            "user_statistics": user_stats,
            "item_statistics": item_stats,
            "genre_distribution": genre_counts,
            "average_rating": float(np.mean(all_ratings)),
            "rating_std": float(np.std(all_ratings))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Something went wrong"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other exceptions"""
    import traceback
    error_details = traceback.format_exc()
    print(f"Unhandled exception: {error_details}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

# Error handlers with proper JSON responses
from fastapi import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions properly"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": exc.detail
        }
    )

@app.exception_handler(HTTPException)
async def fastapi_http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    import traceback
    error_details = traceback.format_exc()
    print("=" * 60)
    print(f"Unhandled Exception: {type(exc).__name__}")
    print(error_details)
    print("=" * 60)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)