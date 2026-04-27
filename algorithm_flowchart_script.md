# Algorithm Structure & Presentation Script

## 1. Flowchart for your Slides
*Tip: You can copy and paste this text directly into your presentation or recreate the boxes using PPT shapes.*

```text
=============================================================================
             STEAM GAME DISCOVERY: ALGORITHM STRUCTURE
=============================================================================

[ Offline Phase ]

  +-----------------------+       +------------------------------------+
  | Steam Games Dataset   |       | SentenceTransformer Model          |
  | (HuggingFace)         |       | (Text -> Semantic Vector mapping)  |
  +-----------+-----------+       +-----------------+------------------+
              |                                     |
              v                                     v
  +-----------------------+       +------------------------------------+
  |   Data Preprocessing  |------>|  Dataset Vectorization (Embedder)  |
  | (Clean fields, tags)  |       |  Output: (N, D) shape Numpy Matrix |
  +-----------------------+       +-----------------+------------------+
                                                    |
         +------------------------------------------+-----------------------+
         |                                                                  |
         v                                                                  v
+-----------------------------+                               +-----------------------------+
|   Custom K-Means++ Engine   |                               |  Vector Search Space Matrix  |
|  (NumPy From-scratch Math)  |                               |    (Normalized Dataset)      |
+--------+--------------------+                               +-------------+---------------+
         |                                                                  |
         v                                                                  |
[ UI Visualization & Map ] <------------------------------------------------+


=============================================================================
[ Online Phase / Retrieval Workflow ]

            [ User Input: "A relaxing farming sim" ]
                              |
                              v
                +---------------------------+       
                | Intelligent Query Builder |   <-   (Expands "relaxing"  
                | (Adds robust synonyms)    |       "cozy", "calm", etc)
                +-------------+-------------+
                              |
                              v
                  +-----------------------+        [ Optional: Dealbreaker ]
                  |   Query Embedding     |        [ Input: "Turn-based"   ]
                  |      (Positive)       |                 |
                  +-----------+-----------+                 v
                              |                   +-----------------------+
                              |                   | Dealbreaker Embedding |
                              |                   |      (Negative)       |
                              v                   +-----------+-----------+
                 +--------------------------------------------+-----------+
                 |          Vector Mathematics (Subtraction)              |
                 |      Query_Vec = Pos_Vec  -  (\alpha * Neg_Vec)        |
                 +----------------------------+---------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Batch Cosine Similarity    |  <-- Computes dot product
                              |       (From-scratch)          |      with entire Vector Search Matrix
                              |  Sim = Query_Vec @ Data_Vec.T |      
                              +---------------+---------------+
                                              |               
                                              v                     
                              +-------------------------------+
                              | Numpy Argsort & Top-K Ranking |  <-- Stage 1: Top-60 Candidate Pool
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Cross-Encoder Reranker     |  <-- Stage 2: Precision scoring Query
                              |      (ms-marco-MiniLM)        |      vs Candidate Descriptions
                              +---------------+---------------+
                                              |
                                              v
                               [ Ranked Top-20 Game Results ]
```

---

## 2. Presentation Script (English)

**(When switching to the slide showing this flowchart):**

"Now, let's take a look at the architecture of our algorithm. We designed a dual-phase system consisting of an offline initialization phase and an online retrieval phase.

**[Point to the top half / Offline Phase]**
First, in the offline phase, we ingested the Steam Games dataset from HuggingFace. We used a Sentence Transformer to convert all game descriptions, genres, and tags into high-dimensional semantic vectors. 

This gives us two main outputs: First, our from-scratch **K-Means++ engine** calculates clusters for our visual map. Second, we store the entire dataset natively as a **Vector Search Space Matrix**.

**[Point to the bottom half / Online Phase]**
The online phase is where the magic happens when a user types a query. Let's say a user searches for 'a relaxing farming sim'. Instead of just matching exact keywords, we pass this through an **Intelligent Query Builder** which mathematically expands vague terms—knowing that 'relaxing' is related to 'cozy' and 'calm'.

**[Explain the Vector Subtraction / Dealbreaker part]**
But what if they *never* want to see turn-based games? If they add a dealbreaker, we embed that dealbreaker as a negative vector. Using vector mathematics, we take the search vector and subtract the dealbreaker vector. It literally pushes the query away from games with turn-based mechanics in the geometric space.

Finally, we run our from-scratch **Batch Cosine Similarity** function. By calculating the dot product of our query vector against the entire dataset matrix using NumPy, we can instantly grab a broad pool of the Top 60 candidate games. 

But we don't stop there. To guarantee absolute precision, we pass these 60 games into our **Two-Stage Cross-Encoder Reranker** model. The Cross-Encoder evaluates the deep semantic interaction between the user's exact query strings and each candidate's text description word-by-word, aggressively re-sorting them to output the true closest matches."
