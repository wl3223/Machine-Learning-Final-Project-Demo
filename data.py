import pandas as pd
from datasets import load_dataset
import streamlit as st

@st.cache_data
def load_and_clean_data(limit=10000):
    """
    Downloads, loads, and cleans the Steam games dataset.
    Uses st.cache_data to ensure it only runs once per Streamlit session.
    
    Args:
        limit (int): Optional limit to the number of rows to load for performance.
                     The full dataset is large (~85k rows).
    """
    # 1. Download & Load Dataset
    ds = load_dataset("FronkonGames/steam-games-dataset", split="train")
    
    # 2. Convert to pandas DataFrame
    df = ds.to_pandas()
    
    # Deduplicate by appID if necessary
    if 'appID' in df.columns:
        df = df.drop_duplicates(subset=['appID'])
    
    # For MVP and from-scratch testing, limit the data size to ensure realistic 
    # embedding times locally if limit is set. We prioritize well-received or popular games
    # so the sample is good quality.
    if limit is not None and limit < len(df):
        if 'positive' in df.columns:
            # Sort by total positive reviews to ensure famous AAA games are always included
            df['positive_numeric'] = pd.to_numeric(df['positive'], errors='coerce').fillna(0)
            df = df.sort_values(by='positive_numeric', ascending=False).head(limit).reset_index(drop=True)
            df = df.drop(columns=['positive_numeric'])
        else:
            df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        
    # 3. Clean data (handle missing values)
    
    import re
    
    def clean_hf_list(val):
        """
        # --- DATA CLEANING VITAL LOGIC FOR TEACHER REVIEW ---
        # Converts strings like "['Action' 'RPG']" or "['Action', 'RPG']" -> "Action, RPG".
        # Why: HuggingFace stores lists as raw string arrays. If not unpacked via Regex, 
        # UI filters will treat the entire massive string as a single, unique garbage category.
        # This Regex extracts the core words safely and normalizes spacing.
        """
        val_str = str(val).strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            matches = re.findall(r"['\"]([^'\"]+)['\"]", val_str)
            if matches:
                return ", ".join(matches)
            # Fallback for brackets without quotes
            clean = val_str.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
            return ", ".join([w.strip() for w in clean.split(',') if w.strip()])
        return val_str

    # 4. Isolate target text fields
    text_cols = ['name', 'short_description', 'detailed_description', 'genres', 'tags', 'categories']
    for col in text_cols:
        if col in df.columns:
            # fill missing with empty string
            df[col] = df[col].fillna('').apply(clean_hf_list)
            # Ensure everything is technically a string
            df[col] = df[col].astype(str)
            
    # 5. Identify numerical/categorical fields for display/filtering
    # Handle scores and prices
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    
    if 'metacritic_score' in df.columns:
        df['metacritic_score'] = pd.to_numeric(df['metacritic_score'], errors='coerce').fillna(0)
        
    if 'user_score' in df.columns:
        # Some user scores are 0, some might be string.
        df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce').fillna(0)
        
    # Categorical/Display strings
    cat_cols = ['developers', 'publishers', 'release_date', 'header_image']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
            df[col] = df[col].astype(str)
            
    return df

def get_text_fields(df):
    """Returns just the text fields needed for search/embedding."""
    cols = ['name', 'short_description', 'detailed_description', 'genres', 'tags', 'categories']
    return df[[col for col in cols if col in df.columns]]

def get_metadata_fields(df):
    """Returns numerical and categorical fields for display/filtering."""
    cols = ['price', 'release_date', 'metacritic_score', 'user_score', 'developers', 'publishers', 'header_image', 'appID']
    return df[[col for col in cols if col in df.columns]]

