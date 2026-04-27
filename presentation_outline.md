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
* **Two-Stage NLP Retrieval (Bi-Encoder + Cross-Encoder):**
  * **Stage 1 (High-Recall):** Rapidly scans the 10,000+ vector space using our custom batch Cosine Similarity to find a broad candidate pool.
  * **Stage 2 (High-Precision):** Employs a Deep Learning `Cross-Encoder` model (`ms-marco-MiniLM`) to dynamically score the semantic relationship between the user's query and each candidate's exact text description, reranking them for maximum accuracy.
* **Custom K-Means++ Clustering:**
  * Segmented the game library grouping similar games for discovery.
  * We implemented the K-Means++ initialization strategy, label assignments using squared Euclidean distance, and centroid re-computations completely from scratch using NumPy arrays.

## Slide 4: Evaluation Metrics & Validation (1 minute)
*(How do we know it actually works?)*
* **Clustering Quality:**
  * We built evaluating metrics running from $K=2$ to $15$.
  * We computed **Inertia (Within-cluster sum of squares)** from scratch and validated it against Sklearn's **Silhouette Score** to find the optimal number of game clusters. 
* **Retrieval Performance (Known-Item Search):**
  * We built an evaluation pipeline computing **Mean Reciprocal Rank (MRR)** and Recall metrics.
  * Pipeline: We hide the original game, feed its `detailed_description` into the query as an embedded vector, and measure if our similarity algorithm can rank that exact game #1 (Hits@1, Hits@5). 

## Slide 5: Real-world Demo & Conclusion (0.5 minutes)
* **Live App Showcase Streamlit UI:** Show the interactive Semantic Map scatterplot and run a unique query (e.g., combining two wildly different genres using the "Cross-Genre" midpoint feature).
* **Takeaway:** Vector embeddings outperform rigid keyword databases for creative recommendations. The math operates natively on text meaning, providing a highly scalable approach to modern media discovery.
