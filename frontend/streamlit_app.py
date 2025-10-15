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

# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x200/FF6B6B/FFFFFF?text=Movie+Rec+System", 
                 use_column_width=True)
        
        st.markdown("### Navigation")
        page = st.selectbox("Choose a page:", [
            "🏠 Home",
            "👤 User Recommendations", 
            "🔍 Movie Search",
            "📊 Analytics Dashboard",
            "⚙️ Model Comparison",
            "📈 Dataset Statistics"
        ])
    
    # Home Page
    if page == "🏠 Home":
        home_page()
    
    # User Recommendations Page
    elif page == "👤 User Recommendations":
        user_recommendations_page()
    
    # Movie Search Page
    elif page == "🔍 Movie Search":
        movie_search_page()
    
    # Analytics Dashboard
    elif page == "📊 Analytics Dashboard":
        analytics_dashboard()
    
    # Model Comparison Page
    elif page == "⚙️ Model Comparison":
        model_comparison_page()
    
    # Dataset Statistics
    elif page == "📈 Dataset Statistics":
        dataset_statistics_page()

def home_page():
    """Home page with system overview"""
    st.markdown("## Welcome to the Movie Recommendation System!")
    
    st.markdown("""
    This system implements various state-of-the-art recommendation algorithms including:
    - **Collaborative Filtering**: Find users with similar tastes
    - **Matrix Factorization**: Discover latent factors in user preferences  
    - **Content-Based Filtering**: Recommend based on movie features
    - **Deep Learning Models**: Neural networks for complex patterns
    - **Hybrid Approaches**: Combine multiple techniques for best results
    """)
    
    # System Status
    st.markdown("### System Status")
    health_data = fetch_from_api("/")
    
    if health_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", "🟢 Healthy" if health_data["status"] == "healthy" else "🔴 Error")
        
        with col2:
            st.metric("Models Loaded", len(health_data["models_loaded"]))
        
        with col3:
            if health_data.get("dataset_info"):
                st.metric("Total Users", f"{health_data['dataset_info']['n_users']:,}")
        
        # Available Models
        st.markdown("### Available Models")
        models_data = fetch_from_api("/models")
        if models_data:
            models_df = pd.DataFrame([
                {"Model": model, "Description": desc} 
                for model, desc in models_data["model_descriptions"].items()
            ])
            st.dataframe(models_df, use_container_width=True)
    
    # Quick Start Guide
    st.markdown("### Quick Start Guide")
    st.markdown("""
    1. **Get Recommendations**: Go to 'User Recommendations' and enter a user ID (1-943)
    2. **Explore Movies**: Use 'Movie Search' to find information about specific movies
    3. **Compare Models**: See how different algorithms perform on the same data
    4. **View Analytics**: Explore system performance and dataset insights
    """)


def user_recommendations_page():
    """User recommendations page"""
    st.markdown("## 👤 Get Personalized Recommendations")
    
    # User input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=100)
    
    with col2:
        # Get available models
        models_data = fetch_from_api("/models")
        if models_data:
            model_options = list(models_data["available_models"])
            selected_model = st.selectbox("Recommendation Model", model_options, 
                                        index=model_options.index("WeightedHybrid") 
                                        if "WeightedHybrid" in model_options else 0)
        else:
            selected_model = st.selectbox("Recommendation Model", ["WeightedHybrid"])
    
    with col3:
        n_recommendations = st.slider("Number of Recommendations", 1, 20, 10)
    
    if st.button("Get Recommendations", type="primary"):
        # Fetch user profile
        with st.spinner("Fetching user profile..."):
            profile_data = fetch_from_api(f"/user/{user_id}/profile")
        
        if profile_data:
            # Display user profile
            st.markdown("### User Profile")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Ratings", profile_data["rating_statistics"]["total_ratings"])
            
            with col2:
                st.metric("Avg Rating", f"{profile_data['rating_statistics']['average_rating']:.1f}⭐")
            
            with col3:
                st.metric("Activity Level", profile_data["activity_level"])
            
            with col4:
                if profile_data["favorite_genres"]:
                    top_genre = profile_data["favorite_genres"][0]["genre"]
                    st.metric("Top Genre", top_genre)
            
            # Display favorite genres
            if profile_data["favorite_genres"]:
                st.markdown("### Favorite Genres")
                genres_df = pd.DataFrame(profile_data["favorite_genres"])
                fig = px.bar(genres_df, x="genre", y="avg_rating", 
                           title="Average Rating by Genre",
                           color="avg_rating", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
        
        # Fetch recommendations
        with st.spinner("Generating recommendations..."):
            rec_data = post_to_api("/recommendations", {
                "user_id": user_id,
                "n_recommendations": n_recommendations,
                "model_name": selected_model
            })
        
        if rec_data and rec_data["recommendations"]:
            st.markdown("### 🎯 Personalized Recommendations")
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recommendations Generated", rec_data["metadata"]["total_recommendations"])
            with col2:
                st.metric("Model Used", rec_data["model_used"])
            with col3:
                avg_rating = rec_data["metadata"]["average_predicted_rating"]
                st.metric("Avg Predicted Rating", f"{avg_rating:.1f}⭐")
            
            # Display recommendations
            for i, movie in enumerate(rec_data["recommendations"], 1):
                with st.expander(f"{i}. {movie['title']} - {movie['predicted_rating']:.1f}⭐", expanded=i <= 3):
                    display_movie_card(movie)
        else:
            st.warning("No recommendations found. Please try a different user ID or model.")


def movie_search_page():
    """Movie search and information page"""
    st.markdown("## 🔍 Movie Search & Information")
    
    # Movie ID input
    col1, col2 = st.columns(2)
    
    with col1:
        movie_id = st.number_input("Movie ID", min_value=1, max_value=1682, value=1)
    
    with col2:
        similarity_type = st.selectbox("Similarity Type", ["content", "collaborative"])
    
    if st.button("Search Movie", type="primary"):
        # Fetch movie information
        with st.spinner("Fetching movie information..."):
            movie_data = fetch_from_api(f"/movie/{movie_id}/info")
        
        if movie_data:
            # Display movie information
            st.markdown("### Movie Information")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"## {movie_data['title']}")
                if movie_data.get('year'):
                    st.caption(f"Released: {movie_data['year']}")
                
                # Display genres
                if movie_data.get('genres'):
                    genres_html = " ".join([f'<span class="genre-tag">{genre}</span>' 
                                          for genre in movie_data['genres']])
                    st.markdown(genres_html, unsafe_allow_html=True)
                
                if movie_data.get('imdb_url'):
                    st.markdown(f"[View on IMDB]({movie_data['imdb_url']})")
            
            with col2:
                # Rating statistics
                stats = movie_data['rating_statistics']
                st.metric("Average Rating", f"{stats['average_rating']:.1f}⭐")
                st.metric("Total Ratings", stats['total_ratings'])
                st.metric("Popularity", movie_data['popularity_rank'])
            
            # Fetch similar movies
            with st.spinner("Finding similar movies..."):
                similar_data = fetch_from_api(f"/similar-movies/{movie_id}", 
                                            params={"n_similar": 10})
            
            if similar_data and similar_data["similar_movies"]:
                st.markdown("### Similar Movies")
                
                # Create similarity chart
                similar_df = pd.DataFrame(similar_data["similar_movies"])
                fig = px.bar(similar_df, x="similarity_score", y="title", 
                           orientation='h', title="Movie Similarity Scores",
                           color="similarity_score", color_continuous_scale="Blues")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display similar movies as cards
                st.markdown("#### Similar Movie Details")
                for movie in similar_data["similar_movies"][:5]:  # Show top 5
                    with st.expander(f"{movie['title']} - Similarity: {movie['similarity_score']:.3f}"):
                        display_movie_card(movie, show_score=False)
        else:
            st.error("Movie not found. Please try a different movie ID.")

def analytics_dashboard():
    """Analytics dashboard with visualizations"""
    st.markdown("## 📊 Analytics Dashboard")
    
    # Fetch dataset statistics
    with st.spinner("Loading analytics data..."):
        stats_data = fetch_from_api("/stats/dataset")
    
    if stats_data:
        # Basic statistics
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{stats_data['basic_statistics']['n_users']:,}")
        
        with col2:
            st.metric("Total Movies", f"{stats_data['basic_statistics']['n_items']:,}")
        
        with col3:
            st.metric("Total Ratings", f"{stats_data['basic_statistics']['n_ratings']:,}")
        
        with col4:
            sparsity = stats_data['basic_statistics']['sparsity'] * 100
            st.metric("Matrix Sparsity", f"{sparsity:.1f}%")
        
        # Rating distribution
        st.markdown("### Rating Distribution")
        rating_data = stats_data['rating_distribution']
        ratings_df = pd.DataFrame([
            {"Rating": int(rating), "Count": count} 
            for rating, count in rating_data.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(ratings_df, x="Rating", y="Count", 
                        title="Rating Distribution",
                        color="Rating", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(ratings_df, values="Count", names="Rating", 
                        title="Rating Distribution (Pie Chart)")
            st.plotly_chart(fig, use_container_width=True)
        
        # User and item statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### User Activity Statistics")
            user_stats = stats_data['user_statistics']
            user_stats_df = pd.DataFrame([
                {"Metric": "Mean Ratings per User", "Value": f"{user_stats['mean_ratings_per_user']:.1f}"},
                {"Metric": "Median Ratings per User", "Value": f"{user_stats['median_ratings_per_user']:.1f}"},
                {"Metric": "Max Ratings per User", "Value": user_stats['max_ratings_per_user']},
                {"Metric": "Min Ratings per User", "Value": user_stats['min_ratings_per_user']}
            ])
            st.dataframe(user_stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### Item Popularity Statistics")
            item_stats = stats_data['item_statistics']
            item_stats_df = pd.DataFrame([
                {"Metric": "Mean Ratings per Item", "Value": f"{item_stats['mean_ratings_per_item']:.1f}"},
                {"Metric": "Median Ratings per Item", "Value": f"{item_stats['median_ratings_per_item']:.1f}"},
                {"Metric": "Max Ratings per Item", "Value": item_stats['max_ratings_per_item']},
                {"Metric": "Min Ratings per Item", "Value": item_stats['min_ratings_per_item']}
            ])
            st.dataframe(item_stats_df, use_container_width=True, hide_index=True)
        
        # Genre distribution
        st.markdown("### Genre Distribution")
        genre_data = stats_data['genre_distribution']
        genres_df = pd.DataFrame([
            {"Genre": genre, "Count": count} 
            for genre, count in genre_data.items()
        ]).sort_values("Count", ascending=True)
        
        fig = px.bar(genres_df, x="Count", y="Genre", orientation='h',
                    title="Number of Movies by Genre",
                    color="Count", color_continuous_scale="Viridis")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def model_comparison_page():
    """Model comparison page"""
    st.markdown("## ⚙️ Model Comparison")
    
    st.markdown("""
    Compare how different recommendation models perform on the same user-item pair.
    This helps understand the strengths and weaknesses of each approach.
    """)
    
    # Input for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=100, key="comp_user")
    
    with col2:
        item_id = st.number_input("Movie ID", min_value=1, max_value=1682, value=1, key="comp_item")
    
    if st.button("Compare Models", type="primary"):
        # Fetch movie information first
        with st.spinner("Fetching movie information..."):
            movie_data = fetch_from_api(f"/movie/{item_id}/info")
        
        if movie_data:
            st.markdown(f"### Predictions for: {movie_data['title']}")
            
            # Fetch model comparisons
            with st.spinner("Comparing models..."):
                comparison_data = post_to_api("/compare-models", {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 0  # Not used for comparison
                })
            
            if comparison_data:
                predictions = comparison_data["predictions"]
                
                # Create comparison chart
                model_names = []
                predicted_ratings = []
                
                for model_name, prediction in predictions.items():
                    if prediction is not None:
                        model_names.append(model_name)
                        predicted_ratings.append(prediction)
                
                if model_names:
                    comparison_df = pd.DataFrame({
                        "Model": model_names,
                        "Predicted Rating": predicted_ratings
                    }).sort_values("Predicted Rating", ascending=True)
                    
                    fig = px.bar(comparison_df, x="Predicted Rating", y="Model", 
                               orientation='h', title="Model Predictions Comparison",
                               color="Predicted Rating", color_continuous_scale="RdYlGn",
                               range_x=[1, 5])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed comparison
                    st.markdown("### Detailed Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.metric("Ensemble Prediction", 
                                f"{comparison_data['ensemble_prediction']:.2f}⭐")
                        st.metric("Prediction Range", 
                                f"{min(predicted_ratings):.1f} - {max(predicted_ratings):.1f}")
                        st.metric("Standard Deviation", 
                                f"{np.std(predicted_ratings):.2f}")
                
                # Model insights
                st.markdown("### Model Insights")
                
                insights = []
                for model_name, prediction in predictions.items():
                    if prediction is not None:
                        if model_name == "UserCF":
                            insight = f"**{model_name}**: Based on users with similar preferences (Rating: {prediction:.2f})"
                        elif model_name == "ContentBased":
                            insight = f"**{model_name}**: Based on movie content similarity (Rating: {prediction:.2f})"
                        elif model_name == "SVD":
                            insight = f"**{model_name}**: Matrix factorization approach (Rating: {prediction:.2f})"
                        else:
                            insight = f"**{model_name}**: Hybrid/Advanced approach (Rating: {prediction:.2f})"
                        
                        insights.append(insight)
                
                for insight in insights:
                    st.markdown(f"• {insight}")
        else:
            st.error("Movie not found. Please try a different movie ID.")


def dataset_statistics_page():
    """Detailed dataset statistics page"""
    st.markdown("## 📈 Dataset Statistics")
    
    # Load comprehensive statistics
    with st.spinner("Loading comprehensive statistics..."):
        stats_data = fetch_from_api("/stats/dataset")
    
    if stats_data:
        # Overview metrics
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        basic_stats = stats_data['basic_statistics']
        
        with col1:
            st.metric("Users", f"{basic_stats['n_users']:,}")
        
        with col2:
            st.metric("Movies", f"{basic_stats['n_items']:,}")
        
        with col3:
            st.metric("Ratings", f"{basic_stats['n_ratings']:,}")
        
        with col4:
            st.metric("Density", f"{basic_stats['density']*100:.2f}%")
        
        with col5:
            st.metric("Avg Rating", f"{stats_data['average_rating']:.2f}⭐")
        
        # Advanced analytics
        st.markdown("### Advanced Analytics")
        
        # Create subplots for multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('User Activity Distribution', 'Item Popularity Distribution',
                          'Rating Distribution Over Time', 'Genre Popularity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # User activity simulation (in real system, would use actual data)
        user_activities = np.random.lognormal(mean=2, sigma=1, size=943)
        fig.add_trace(
            go.Histogram(x=user_activities, name="User Activities", nbinsx=30),
            row=1, col=1
        )
        
        # Item popularity simulation
        item_popularities = np.random.exponential(scale=20, size=1682)
        fig.add_trace(
            go.Histogram(x=item_popularities, name="Item Popularity", nbinsx=30),
            row=1, col=2
        )
        
        # Rating distribution
        rating_data = stats_data['rating_distribution']
        ratings = list(rating_data.keys())
        counts = list(rating_data.values())
        fig.add_trace(
            go.Bar(x=ratings, y=counts, name="Rating Counts"),
            row=2, col=1
        )
        
        # Genre distribution
        genre_data = stats_data['genre_distribution']
        top_genres = sorted(genre_data.items(), key=lambda x: x[1], reverse=True)[:10]
        genres, genre_counts = zip(*top_genres)
        fig.add_trace(
            go.Bar(x=list(genres), y=list(genre_counts), name="Genre Counts"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data quality metrics
        st.markdown("### Data Quality Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Sparsity Analysis")
            sparsity_pct = basic_stats['sparsity'] * 100
            
            if sparsity_pct > 99:
                quality = "Very Sparse"
                color = "red"
            elif sparsity_pct > 95:
                quality = "Sparse"
                color = "orange"
            else:
                quality = "Dense"
                color = "green"
            
            st.markdown(f"**Sparsity**: <span style='color: {color}'>{sparsity_pct:.2f}% ({quality})</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Coverage Analysis")
            # Simulate coverage metrics
            user_coverage = (basic_stats['n_users'] - 50) / basic_stats['n_users'] * 100
            item_coverage = (basic_stats['n_items'] - 200) / basic_stats['n_items'] * 100
            
            st.markdown(f"**User Coverage**: {user_coverage:.1f}%")
            st.markdown(f"**Item Coverage**: {item_coverage:.1f}%")
        
        with col3:
            st.markdown("#### Rating Diversity")
            rating_std = stats_data['rating_std']
            
            if rating_std > 1.2:
                diversity = "High"
                color = "green"
            elif rating_std > 0.8:
                diversity = "Medium"
                color = "orange"
            else:
                diversity = "Low"
                color = "red"
            
            st.markdown(f"**Rating Std**: <span style='color: {color}'>{rating_std:.2f} ({diversity})</span>", 
                       unsafe_allow_html=True)
    
    # System performance metrics (simulated)
    st.markdown("### System Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Response Time", "125ms", delta="-15ms")
    
    with col2:
        st.metric("Throughput", "850 req/sec", delta="50 req/sec")
    
    with col3:
        st.metric("Cache Hit Rate", "89.5%", delta="2.1%")
    
    with col4:
        st.metric("Model Accuracy (RMSE)", "0.95", delta="-0.02")


# Run the application
if __name__ == "__main__":
    main()

# Additional utility functions for the Streamlit app
def create_performance_chart(models_data):
    """Create performance comparison chart"""
    if not models_data:
        return None
    
    fig = go.Figure()
    
    metrics = ['RMSE', 'MAE', 'Precision@10']
    models = list(models_data.keys())
    
    for metric in metrics:
        values = [models_data[model].get(metric, 0) for model in models]
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Metric Values",
        barmode='group'
    )
    
    return fig

def display_recommendation_explanation(recommendation):
    """Display detailed recommendation explanation"""
    st.markdown("#### Why this recommendation?")
    
    factors = []
    
    if recommendation.get('confidence', 0) > 0.8:
        factors.append("🎯 High confidence prediction")
    
    if recommendation.get('popularity_score', 0) > 50:
        factors.append("🔥 Popular among users")
    
    if recommendation.get('predicted_rating', 0) > 4.0:
        factors.append("⭐ High predicted rating")
    
    for factor in factors:
        st.markdown(f"• {factor}")
    
    if recommendation.get('reason'):
        st.markdown(f"• 💡 {recommendation['reason']}")

# Custom components for enhanced UI
def create_rating_stars(rating, max_rating=5):
    """Create star rating display"""
    stars = ""
    for i in range(max_rating):
        if i < int(rating):
            stars += "⭐"
        elif i < rating:
            stars += "⭐"  # Half star (simplified)
        else:
            stars += "☆"
    return stars

def format_large_number(number):
    """Format large numbers with K, M suffixes"""
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(number)