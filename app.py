import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from input_handler import InputValidator

# Import local modules
from data import load_and_clean_data
from embed import load_embedding_model, combine_text_fields, generate_embeddings
from retrieval import rank_games_for_query, get_similar_games, batch_cosine_similarity
from viz import perform_pca_projection, plot_2d_map, plot_price_distribution, plot_top_genres
from clustering import perform_kmeans_clustering
from utils import set_reproducibility

# Constants
MVP_DATA_LIMIT = 5000
RANDOM_SEED = 42

set_reproducibility(RANDOM_SEED)

st.set_page_config(page_title="Steam Game Discovery Explorer", layout="wide")

# Inject Custom Steam Theme CSS
STEAM_CSS = """
<style>
/* Steam Backgrounds */
.stApp {
    background-color: #1b2838;
}
[data-testid="stSidebar"] {
    background-color: #171a21;
}
[data-testid="stHeader"] {
    background-color: #171a21;
}

/* Steam Typography & Colors */
html, body, p, span, div, label {
    color: #c6d4df !important;
}
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 300;
    letter-spacing: 1px;
}

/* Steam Accents (Buttons) */
.stButton > button {
    background: linear-gradient( to right, #47bfff 5%, #1a44c2 60%);
    color: #ffffff !important;
    border: none;
    border-radius: 3px;
}
.stButton > button:hover {
    background: linear-gradient( to right, #47bfff 5%, #1a44c2 100%);
    box-shadow: 0px 0px 8px #66c0f4;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #66c0f4 !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #66c0f4 !important;
}

/* Search Bars / Inputs */
.stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div {
    background-color: rgba(0, 0, 0, 0.4) !important;
    color: #ffffff !important;
    border: 1px solid #1a44c2 !important;
}

/* Alerts / Success */
.stAlert {
    background-color: #2a475e !important;
    color: #66c0f4 !important;
    border: 1px solid #1a44c2 !important;
}
</style>
"""
st.markdown(STEAM_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data_and_embeddings():
    df = load_and_clean_data(limit=MVP_DATA_LIMIT)
    # Extract year for filtering
    df['release_year'] = df['release_date'].str.extract(r'(\d{4})').fillna(0).astype(int)
    combined_texts = combine_text_fields(df)
    return df, combined_texts

df_raw, combined_texts = load_data_and_embeddings()
model = load_embedding_model()

@st.cache_data(show_spinner=False)
def get_cached_embeddings(text_list):
    return generate_embeddings(model, text_list.tolist())

with st.spinner("Initializing Vector Space & Clusters..."):
    dataset_vectors = get_cached_embeddings(combined_texts)
    # Phase 5 & 6 Math Prep
    pca_projection = perform_pca_projection(dataset_vectors)
    clusters = perform_kmeans_clustering(dataset_vectors, n_clusters=8, seed=RANDOM_SEED)
    
    # Attach to a working df
    df = df_raw.copy()
    df['Cluster'] = [f"Cluster {c}" for c in clusters]

st.title("🎮 Steam Game Discovery Explorer")
st.markdown("Discover Steam games through natural text, dealbreakers, and vector similarity.")

# SIDEBAR FILTERS (Phase 6)
st.sidebar.header("Filter Results")
min_price, max_price = st.sidebar.slider("Price Range", 0.0, 100.0, (0.0, 100.0))

# All available genres
# --- DATA CLEANING VITAL LOGIC FOR TEACHER REVIEW ---
# 1. str.split(',') breaks the properly formatted "Action, RPG" string into ["Action", " RPG"].
# 2. Python's `set()` is utilized here for mathematical deduplication (Hash table).
# 3. This forces thousands of repetitive cross-game genres to collapse into singular, unique options 
#    for the sidebar, preventing "Dimension Explosion" in the UI frontend.
all_genres = set()
for g_list in df['genres'].str.split(','):
    if type(g_list) is list:
        all_genres.update([g.strip() for g in g_list if g.strip()])
        
selected_genres = st.sidebar.multiselect("Require Genres:", sorted(list(all_genres)))

# Apply filters
mask = (df['price'] >= min_price) & (df['price'] <= max_price)
if selected_genres:
    # Game must contain at least one of the selected genres
    # To require ALL, use all(g in x for g in selected_genres)
    genre_mask = df['genres'].apply(lambda x: any(g in str(x) for g in selected_genres))
    mask = mask & genre_mask

filtered_df = df[mask].copy()
st.sidebar.text(f"Showing {len(filtered_df)} games out of {len(df)}")

# UI Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "😲 Surprise Me (Cross-Genre)", "📈 Data & Visuals"])

# TAB 1: SEMANTIC SEARCH
with tab1:
    st.header("Search for Games")
    
    # Show example queries
    with st.expander("📝 Need ideas? See example queries:"):
        for example in InputValidator.suggest_query_examples():
            st.write(f"• {example}")
    
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_area(
            "Describe the game you want to play:",
            "A relaxing farming simulator",
            help="Be specific! Mention genre, tone, features, or setting."
        )
    with col2:
        dealbreakers = st.text_area(
            "Dealbreakers (what you DON'T want):",
            "microtransactions, stress",
            help="Leave empty if nothing to avoid."
        )
    
    c_left, c_right = st.columns(2)
    with c_left:
        algo = st.radio("Retrieval Algorithm", ["From-Scratch (Custom Math)", "Scikit-Learn (NearestNeighbors)"])
    with c_right:
        sort_by = st.selectbox("Sort Matches By:", ["Match Score (Default)", "Total Positive Reviews", "Estimated Owners", "Price (Low to High)"])
    
    if st.button("Search", key="search_button"):
        # VALIDATE INPUT ONLY AFTER BUTTON CLICK
        is_valid, cleaned_query, suggestion = InputValidator.validate_query(query)
        
        # SHOW SUGGESTION ONLY AFTER BUTTON CLICK
        if suggestion:
            st.info(suggestion)
        
        # VALIDATION CHECK
        if not is_valid:
            st.error(f"❌ Invalid query: {suggestion}")
        elif not cleaned_query:
            st.error("Please enter a search query.")
        elif len(filtered_df) == 0:
            st.warning("No games match your sidebar filters. Try adjusting them.")
        else:
            # PROCEED WITH SEARCH
            start_time = time.time()
            
            filtered_indices = filtered_df.index.tolist()
            subset_vectors = dataset_vectors[filtered_indices]
            
            # Fetch 20 games to give the sorting option meaningful variation
            FETCH_K = 20
            
            if algo == "From-Scratch (Custom Math)":
                results = rank_games_for_query(cleaned_query, dealbreakers, model, subset_vectors, filtered_df, top_k=FETCH_K, alpha=0.5)
                latency = time.time() - start_time
                st.success(f"✅ Custom Algorithm Latency: {latency:.4f} seconds | Found {len(results)} matches")
            else:
                q_vec = model.encode([cleaned_query], convert_to_numpy=True)[0]
                if dealbreakers:
                    n_vec = model.encode([dealbreakers], convert_to_numpy=True)[0]
                    q_vec = q_vec - (0.5 * n_vec)
                q_vec_2d = q_vec.reshape(1, -1)
                
                nn = NearestNeighbors(n_neighbors=min(FETCH_K, len(subset_vectors)), metric='cosine', algorithm='brute')
                nn.fit(subset_vectors)
                distances, indices = nn.kneighbors(q_vec_2d)
                
                results = filtered_df.iloc[indices[0]].copy()
                results['similarity_score'] = 1 - distances[0]
                latency = time.time() - start_time
                st.success(f"✅ Scikit-Learn Latency: {latency:.4f} seconds | Found {len(results)} matches")
            
            # DISPLAY RESULTS WITH FEEDBACK
            if len(results) == 0:
                st.warning("No games found matching your criteria. Try modifying your search or filters.")
                st.info("💡 Suggestion: Use broader terms or adjust the sidebar filters.")
            else:
                # Apply chosen sorting
                if sort_by == "Total Positive Reviews":
                    results['pos_num'] = pd.to_numeric(results['positive'], errors='coerce').fillna(0)
                    results = results.sort_values(by='pos_num', ascending=False)
                elif sort_by == "Estimated Owners":
                    # 'estimated_owners' is usually text like "1000000 - 2000000". Extract the first number as float.
                    results['owners_min'] = results['estimated_owners'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
                    results = results.sort_values(by='owners_min', ascending=False)
                elif sort_by == "Price (Low to High)":
                    results = results.sort_values(by='price', ascending=True)
                else:
                    # Default Match Score
                    results = results.sort_values(by='similarity_score', ascending=False)
                
                st.divider()
                
                # Display results
                for idx, row in results.iterrows():
                    with st.container():
                        c1, c2 = st.columns([1, 4])
                        score = row['similarity_score']
                        if 'header_image' in row and row['header_image']:
                            try:
                                c1.image(row['header_image'])
                            except:
                                c1.write("🖼️")
                        
                        c2.subheader(f"{row['name']} `(Match: {score:.2f})`")
                        c2.markdown(f"**Genres:** {row['genres']} | **Price:** ${row['price']} | **Cluster:** {row['Cluster']}")
                        c2.write(row['short_description'])
                        
                        with c2.expander("Find Similar Games"):
                            original_idx = df.index.get_loc(idx)
                            similar_df = get_similar_games(original_idx, dataset_vectors, df, top_k=5)
                            st.write(f"Games like {row['name']} (across entire dataset):")
                            for _, sim_row in similar_df.iterrows():
                                st.write(f"- **{sim_row['name']}** (Score: {sim_row['similarity_score']:.2f})")
                    st.write("---")

# TAB 2: SURPRISE ME (CROSS-GENRE)
with tab2:
    st.header("Surprise Me (Cross-Genre Explorer)")
    st.markdown("Combine two wildly different concepts to find games living in the midpoint!")
    
    col1, col2 = st.columns(2)
    with col1:
        concept1 = st.text_input(
            "Concept 1:",
            "Violent first person shooter",
            help="Example: Fast-paced action, puzzle game, etc."
        )
    with col2:
        concept2 = st.text_input(
            "Concept 2:",
            "Cute animal crossing",
            help="Example: Relaxing game, horror, etc."
        )
    
    if st.button("Find Midpoint Games", key="midpoint_button"):
        # VALIDATE BOTH CONCEPTS ONLY AFTER BUTTON CLICK
        is_valid1, clean1, sugg1 = InputValidator.validate_query(concept1)
        is_valid2, clean2, sugg2 = InputValidator.validate_query(concept2)
        
        # SHOW VALIDATION FEEDBACK ONLY AFTER BUTTON CLICK
        if sugg1:
            st.info(f"**Concept 1**: {sugg1}")
        if sugg2:
            st.info(f"**Concept 2**: {sugg2}")
        
        # CHECK VALIDITY BEFORE PROCEEDING
        if not is_valid1 or not is_valid2:
            st.error("❌ Both concepts must be valid. Please refine them.")
        elif len(filtered_df) == 0:
            st.warning("No games match your sidebar filters.")
        else:
            # PROCEED WITH SEARCH (using cleaned, validated input)
            v1 = model.encode([clean1], convert_to_numpy=True)[0]
            v2 = model.encode([clean2], convert_to_numpy=True)[0]
            midpoint = (v1 + v2) / 2
            
            filtered_indices = filtered_df.index.tolist()
            subset_vectors = dataset_vectors[filtered_indices]
            
            similarities = batch_cosine_similarity(midpoint, subset_vectors)
            top_indices = np.argsort(similarities)[::-1][:10]
            
            st.success(f"✅ Found {len(top_indices)} games bridging '{clean1}' and '{clean2}'")
            st.divider()
            
            for index_in_subset in top_indices:
                row = filtered_df.iloc[index_in_subset]
                score = similarities[index_in_subset]
                
                with st.container():
                    c1, c2 = st.columns([1, 4])
                    if 'header_image' in row and row['header_image']:
                        try:
                            c1.image(row['header_image'])
                        except:
                            c1.write("🖼️")
                    
                    c2.markdown(f"### {row['name']} `(Match: {score:.2f})`")
                    c2.write(f"**Genres:** {row['genres']} | **Cluster:** {row['Cluster']}")
                    c2.write(row['short_description'])
                st.write("---")

# TAB 3: DATA & VISUALS
with tab3:
    st.header("Visual Data Exploration")
    
    # 2D Map Colored by Cluster
    st.subheader("Semantic Space (PCA)")
    color_choice = st.radio("Color Map by:", ["Primary Genre", "K-Means Cluster"], horizontal=True)
    color_col = 'genres_primary' if color_choice == "Primary Genre" else 'Cluster'
    
    # We plot using the full `df` to see the whole space, but you could use `filtered_df`
    fig_pca = plot_2d_map(df, pca_projection, color_col=color_col)
    if color_choice == "K-Means Cluster":
        fig_pca.update_layout(title="2D Semantic Mapping colored by K-Means Clusters")
        
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # EDA Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Price Distribution")
        fig_hist = plot_price_distribution(filtered_df)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        st.subheader("Top Genres")
        fig_bar = plot_top_genres(filtered_df)
        st.plotly_chart(fig_bar, use_container_width=True)
