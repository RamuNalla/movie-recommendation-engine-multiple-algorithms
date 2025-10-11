from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append('../src')

from data.data_loader import MovieLensDataLoader
from models.hybrid import WeightedHybrid

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

# Pydantic models for API
@asynccontextmanager
async def lifespan(app):
    global models_dict, user_item_matrix, movies_df, preprocessor, model_info
    try:
        print("Loading data...")
        loader = MovieLensDataLoader('data/raw/ml-100k/')
        ratings, movies, users = loader.load_all_data()
        movies_df = movies
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        user_item_matrix, matrix_info = preprocessor.create_user_item_matrix(ratings)
        print("Loading models...")
        from models.collaborative_filtering import CollaborativeFiltering
        from models.matrix_factorization import SVDRecommender
        from models.content_based import ContentBasedRecommender
        train_data, _ = loader.get_train_test_split()
        train_ratings = train_data.copy()
        train_ratings['user_idx'] = preprocessor.user_encoder.transform(train_ratings['user_id'])
        train_ratings['item_idx'] = preprocessor.item_encoder.transform(train_ratings['item_id'])
        train_matrix = np.zeros_like(user_item_matrix)
        for _, row in train_ratings.iterrows():
            train_matrix[int(row['user_idx']), int(row['item_idx'])] = row['rating']
        cf_model = CollaborativeFiltering(similarity_metric='cosine', n_neighbors=50)
        cf_model.fit(train_matrix)
        svd_model = SVDRecommender(n_factors=50, random_state=42)
        svd_model.fit(train_matrix)
        content_model = ContentBasedRecommender()
        content_model.fit(movies, train_ratings)
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
    yield
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