from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

from data.data_loader import MovieLensDataLoader
from models.hybrid import WeightedHybrid

app = FastAPI(
    title="Movie Recommendation API",
    description="A comprehensive movie recommendation system using various ML algorithms",
    version="1.0.0"
)

