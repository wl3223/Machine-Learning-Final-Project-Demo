import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st

@st.cache_data(show_spinner=False)
def perform_pca_projection(_dataset_vectors):
    """
    Reduces dataset dense vectors to 2D using PCA for visualization.
    Uses an underscore on the vectors variable to avoid Streamlit full hashing.
    """
    pca = PCA(n_components=2)
    projection = pca.fit_transform(_dataset_vectors)
    return projection

def plot_2d_map(df, projection, color_col='genres_primary'):
    """
    Creates a 2D scatter plot map of games.
    """
    import re
    plot_df = df.copy()
    plot_df['PC1'] = projection[:, 0]
    plot_df['PC2'] = projection[:, 1]
    
    if color_col == 'genres_primary':
        # Safely extract primary genre since data.py now natively outputs clean "Action, RPG" strings
        plot_df['genres_primary'] = plot_df['genres'].apply(lambda x: str(x).split(',')[0].strip() if x else 'Unknown')
        
        # Limit colors to the top 10 most frequent to avoid chaotic legends
        top_genres = plot_df['genres_primary'].value_counts().nlargest(10).index
        plot_df.loc[~plot_df['genres_primary'].isin(top_genres), 'genres_primary'] = 'Other'
    
    fig = px.scatter(
        plot_df, 
        x='PC1', 
        y='PC2', 
        color=color_col,
        hover_name='name',
        hover_data=['price', 'metacritic_score', 'release_date'],
        title="2D Semantic Mapping of Steam Games (PCA)",
        opacity=0.7,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Light24
    )
    fig.update_layout(height=600)
    return fig

def plot_price_distribution(df):
    """Plots a histogram of game prices."""
    # Filter out extreme outliers for a readable distribution
    plot_df = df[df['price'] <= 100].copy()
    fig = px.histogram(
        plot_df, 
        x='price', 
        nbins=50,
        title="Price Distribution (<= $100)",
        labels={'price': 'Price ($)'},
        template="plotly_dark",
        color_discrete_sequence=['#1f77b4']
    )
    return fig

def plot_top_genres(df):
    """Plots a bar chart of the most common genres."""
    # Explode the comma-separated genres
    genres = df['genres'].str.split(',').explode().str.strip()
    genres = genres[genres != '']
    top_genres = genres.value_counts().nlargest(15)
    
    fig = px.bar(
        x=top_genres.index, 
        y=top_genres.values,
        title="Top 15 Most Common Genres",
        labels={'x': 'Genre', 'y': 'Number of Games'},
        template="plotly_dark",
        color_discrete_sequence=['#ff7f0e']
    )
    return fig

