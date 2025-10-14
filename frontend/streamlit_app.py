import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .genre-tag {
        background-color: #4ECDC4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_from_api(endpoint, params=None):
    """Fetch data from API with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
    
def post_to_api(endpoint, data):
    """Post data to API"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
    

def display_movie_card(movie_data, show_score=True):
    """Display a movie recommendation card"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{movie_data['title']}**")
            if movie_data.get('year'):
                st.caption(f"Year: {movie_data['year']}")
            
            # Display genres as tags
            if movie_data.get('genres'):
                genres_html = " ".join([f'<span class="genre-tag">{genre}</span>' 
                                      for genre in movie_data['genres']])
                st.markdown(genres_html, unsafe_allow_html=True)
            
            # Display reason if available
            if movie_data.get('reason'):
                st.caption(f"💡 {movie_data['reason']}")
        
        with col2:
            if show_score and movie_data.get('predicted_rating'):
                score = movie_data['predicted_rating']
                color = "green" if score >= 4.0 else "orange" if score >= 3.5 else "red"
                st.markdown(f"<h3 style='color: {color}; text-align: center;'>{score:.1f}⭐</h3>", 
                           unsafe_allow_html=True)
            
            if movie_data.get('confidence'):
                conf_pct = movie_data['confidence'] * 100
                st.progress(movie_data['confidence'])
                st.caption(f"Confidence: {conf_pct:.1f}%")
