# Steam Game Discovery: A Semantic Vector-Based Approach
**5-Minute Final Project Presentation Outline**

## Slide 1: Introduction & The Problem (1 minute)
* **The Problem:** Traditional game marketplaces (like Steam) rely entirely on boolean filters (e.g., matching exact "Genres" or "Tags"). If you want "a relaxing game where I don't have to think," traditional databases fail.
* **Our Solution:** A semantic, vector-based retrieval system. We mapped the Steam dataset into a mathematical space where games with similar "vibes" and mechanics live close to each other.
* **Key Features:** Natural language search, "Dealbreaker" negative vector subtraction, and discovering games mathematically rather than categorically.

## Slide 2: Data Pipeline & Embedding Strategy (1 minute)
* **Dataset:** Utilized the `FronkonGames/steam-games-dataset` from HuggingFace, filtering and cleaning relevant text columns (genres, tags, descriptions).
* **Text Embeddings (Sentence Transformers):** We generated semantic vectors. This allows the model to understand that "scary" and "horror" belong together, even if the exact word isn't in the game description.
* **Intelligent Query Expansion:** When a user types a vague query like "relaxing", our algorithm automatically expands it using hidden synonyms ("cozy", "calm", "laid-back") to build a robust query vector before searching.

## Slide 3: Core Algorithm Implementation Details (1.5 minutes)
*(Highlight that the core math was written from scratch!)*
* **Custom Cosine Similarity Retrieval:** 
  * Instead of library black-boxes, we implemented vector normalization ($v / ||v||$) and batch cosine similarity via dot products in NumPy.
  * **Vector Addition/Subtraction:** To handle "dealbreakers" (e.g., negative queries), we encode the negative string, scale it by a weight ($\alpha$), and mathematically subtract it from the positive query vector before retrieving.
* **Ultimate Three-Stage Hybrid NLP Architecture:**
  * **Stage 1 (High-Recall Hybrid Search):** Processes queries simultaneously through a Deep Learning `Bi-Encoder` (for semantic context) and a mathematical `TF-IDF Matrix` (for precise keyword hits).
  * **Stage 2 (Reciprocal Rank Fusion - RRF):** Unifies these two distinct rankings using the $RRF = \frac{1}{60 + R_{dense}} + \frac{1}{60 + R_{sparse}}$ algorithm to guarantee both meaning and precision are prioritized equally.
  * **Stage 3 (High-Precision DL):** Employs a Sequence Classifying `Cross-Encoder` model (`ms-marco-MiniLM`) to dynamically output the final confidence probabilities by deeply analyzing the query vs the top candidate descriptions.
* **Custom K-Means++ Clustering:**
  * Segmented the game library grouping similar games for discovery.
  * We implemented the K-Means++ initialization strategy, label assignments using squared Euclidean distance, and centroid re-computations completely from scratch using NumPy arrays.

## Slide 4: Evaluation Metrics & Validation (1 minute)
*(How do we know it actually works?)*
* **Clustering Quality:**
  * We built evaluating metrics running from $K=2$ to $15$.
  * We computed **Inertia (Within-cluster sum of squares)** from scratch and validated it against Sklearn's **Silhouette Score** to find the optimal number of game clusters. 
* **Retrieval Performance & Global Pre-loading:**
  * To evaluate our Ultimate Three-Stage Pipeline accurately without lagging the frontend, we built a Global **Pre-Loaded Zero-Latency Benchmark**.
  * During website initialization, the server automatically executes the intensive Cross-Encoder validation sequence on N=30 randomized target samples and caches the WCSS / MRR / Recall numbers permanently for instantaneous frontend metrics.

## Slide 5: Real-world Demo & Conclusion (0.5 minutes)
* **Live App Showcase Streamlit UI:** Show the interactive Semantic Map scatterplot and run a unique query (e.g., combining two wildly different genres using the "Cross-Genre" midpoint feature).
* **Takeaway:** Vector embeddings outperform rigid keyword databases for creative recommendations. The math operates natively on text meaning, providing a highly scalable approach to modern media discovery.
