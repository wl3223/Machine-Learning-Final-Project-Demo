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
        color_discrete_sequence=px.colors.qualitative.Light24,
        category_orders={color_col: sorted(plot_df[color_col].astype(str).unique())}
    )
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c6d4df',
        title_font_color='#ffffff'
    )
    return fig

def plot_price_pie(df):
    """Plots a pie chart of Free vs Paid games."""
    free_count = (df['price'] == 0).sum()
    paid_count = (df['price'] > 0).sum()
    
    fig = px.pie(
        names=["Free to Play ($0)", "Paid"],
        values=[free_count, paid_count],
        title="Free vs Paid Proportion",
        template="plotly_dark",
        color_discrete_sequence=['#2ca02c', '#1f77b4']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # Make pie chart visually match heights of other charts
    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c6d4df',
        title_font_color='#ffffff'
    )
    return fig

def plot_price_distribution(df):
    """Plots a histogram of purely paid game prices."""
    # Filter out extreme outliers and Free games
    plot_df = df[(df['price'] > 0) & (df['price'] <= 100)].copy()
    fig = px.histogram(
        plot_df, 
        x='price', 
        nbins=40,
        title="Paid Price Distribution (<= $100)",
        labels={'price': 'Price ($)'},
        template="plotly_dark",
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c6d4df',
        title_font_color='#ffffff'
    )
    return fig

def plot_top_genres(df):
    """Plots a horizontal bar chart of the most common genres."""
    # Explode the comma-separated genres
    genres = df['genres'].str.split(',').explode().str.strip()
    genres = genres[genres != '']
    top_genres = genres.value_counts().nlargest(15)
    
    # Sort ascending so the largest value appears at the VERY TOP of the horizontal chart
    top_genres = top_genres.sort_values(ascending=True)
    
    plot_df = pd.DataFrame({'Genre': top_genres.index, 'Count': top_genres.values})
    
    fig = px.bar(
        plot_df, 
        x='Count', 
        y='Genre',
        orientation='h',
        title="Top 15 Most Common Genres",
        labels={'Count': 'Number of Games', 'Genre': ''},
        template="plotly_dark",
        color='Count',
        color_continuous_scale='Sunsetdark'
    )
    fig.update_layout(
        height=500, 
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c6d4df',
        title_font_color='#ffffff'
    )
    return fig

def plot_elbow_silhouette(metrics_df):
    """Plots Inertia (Elbow) and Silhouette Score against K values."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=metrics_df['k'], y=metrics_df['inertia'], name="Inertia (WCSS)", line=dict(color="#47bfff", width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=metrics_df['k'], y=metrics_df['silhouette'], name="Silhouette Score", line=dict(color="#ff9900", width=3)),
        secondary_y=True,
    )
    
    fig.update_layout(
        title="Elbow Method & Silhouette Score vs K",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c6d4df',
        title_font_color='#ffffff',
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_xaxes(title_text="Number of Clusters (k)", tickmode='linear', tick0=2, dtick=1)
    fig.update_yaxes(title_text="<b>Inertia</b> (Lower is Better)", color="#47bfff", secondary_y=False)
    fig.update_yaxes(title_text="<b>Silhouette Score</b> (Higher is Better)", color="#ff9900", secondary_y=True)
    
    return fig
