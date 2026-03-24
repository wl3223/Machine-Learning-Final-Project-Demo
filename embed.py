import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import os

@st.cache_resource
def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads and caches the SentenceTransformer model.
    'all-MiniLM-L6-v2' is chosen for a fast, high-quality MVP.
    """
    return SentenceTransformer(model_name)

def combine_text_fields(df):
    """
    Combines text fields (title, genres, tags, description) into one dense string per game.
    """
    def ensure_str(val):
        return str(val) if val else ""

    combined = (
        "Title: " + df['name'].apply(ensure_str) + ". " +
        "Genres: " + df['genres'].apply(ensure_str) + ". " +
        "Tags: " + df['tags'].apply(ensure_str) + ". " +
        "Categories: " + df['categories'].apply(ensure_str) + ". " +
        "Description: " + df['short_description'].apply(ensure_str)
    )
    return combined

@st.cache_data(show_spinner=False)
def generate_embeddings(_model, text_list):
    """
    Generates embeddings for a list of text strings.
    _model has an underscore to prevent Streamlit from hashing the model object.
    
    Returns:
        np.ndarray: Matrix of embeddings.
    """
    return _model.encode(text_list, show_progress_bar=True, convert_to_numpy=True)

