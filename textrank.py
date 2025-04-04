import json
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np

# Ensure the Punkt tokenizer is available
# nltk.download('punkt')

def flatten_sections(sections):
    """Flatten a list of section lists into one list of sentences."""
    return [sentence for section in sections for sentence in section]

def text_rank(sentences, model, top_k=20, similarity_threshold=0.1):
    """
    Apply TextRank using cosine similarity on a list of sentences with pruning.
    
    Args:
        sentences (List[str]): List of sentences.
        model (SentenceTransformer): Preloaded sentence embedding model.
        top_k (int): Number of top sentences to select.
        similarity_threshold (float): Prune similarity values below this threshold.
        
    Returns:
        List[str]: Top ranked sentences, sorted in their original order.
    """
    # Step 1: Compute sentence embeddings.
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Step 2: Compute the cosine similarity matrix.
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    
    # Step 3: Prune the similarity matrix by setting values below threshold to 0.
    pruned_sim_matrix = np.where(sim_matrix >= similarity_threshold, sim_matrix, 0)
    
    # Step 4: Build graph using the pruned similarity matrix.
    graph = nx.from_numpy_array(pruned_sim_matrix)
    
    # Step 5: Run PageRank.
    scores = nx.pagerank(graph, max_iter=10000, tol=1e-06)
    
    # Step 6: Rank sentences by their PageRank scores.
    # Include original index for later ordering.
    ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
    
    # Step 7: Select top_k sentences based on score.
    top_k_ranked = ranked_sentences[:top_k]
    
    # Step 8: Sort the selected sentences by their original index (i.e., their order in the article).
    top_k_sorted = sorted(top_k_ranked, key=lambda x: x[1])
    
    # Return only the sentence strings.
    return [s for score, i, s in top_k_sorted]

def main():
    # Define the path for the test subset file.
    input_path = Path("test/plos_test_subset.json")
    
    # Load the JSON file containing the subset of articles.
    with open(input_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Load the domain-specific sentence embedding model.
    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    
    # Process each article.
    for article in articles:
        article_id = article.get("id", "N/A")
        title = article.get("title", "No Title")
        sections = article.get("sections", [])
        
        # Flatten the sections into a list of sentences.
        sentences = flatten_sections(sections)
        if not sentences:
            print(f"Article {article_id} has no sentences; skipping.")
            continue
        
        # Apply the modified TextRank method with pruning and return top 20, then sort by original order.
        top_sentences = text_rank(sentences, model, top_k=20, similarity_threshold=0.1)
        print(f"\nArticle ID: {article_id}")
        print(f"Title: {title}")
        print("Top 20 TextRank Sentences (in original order):")
        for idx, sent in enumerate(top_sentences, start=1):
            print(f"{idx}. {sent}")
        print("-" * 80)

if __name__ == "__main__":
    main()
