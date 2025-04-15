import json
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np

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
    
    # Step 3: Prune the similarity matrix by setting values below the threshold to 0.
    pruned_sim_matrix = np.where(sim_matrix >= similarity_threshold, sim_matrix, 0)
    
    # Step 4: Build graph using the pruned similarity matrix.
    graph = nx.from_numpy_array(pruned_sim_matrix)
    
    # Step 5: Run PageRank.
    scores = nx.pagerank(graph, max_iter=10000, tol=1e-06)
    
    # Step 6: Include original index for later ordering.
    ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
    
    # Step 7: Select the top_k sentences based on PageRank score.
    top_k_ranked = ranked_sentences[:top_k]
    
    # Step 8: Sort the selected sentences by their original index.
    top_k_sorted = sorted(top_k_ranked, key=lambda x: x[1])
    
    # Return only the sentence strings.
    return [s for score, i, s in top_k_sorted]


def main():
    # Define input and output paths.
    input_path = Path("../plos/test.json")
    output_path = "../plos/test_textrank.json"
    
    # Load the JSON file containing the subset of articles.
    with open(input_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Load the domain-specific sentence embedding model.
    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

    count = 0
    # Process each article in the input file.
    for article in articles:
        sections = article.get("sections", [])
        sentences = flatten_sections(sections)
        if not sentences:
            article["top_k"] = []
            continue
        
        # Compute top_k ranked sentences.
        top_sentences = text_rank(sentences, model, top_k=20, similarity_threshold=0.1)
        article["top_k"] = top_sentences
        count += 1
        if count % 10 == 0:
            print(f"Processing {count} articles")

    # Save the updated articles list to the output file.
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(articles, f_out, indent=2, ensure_ascii=False)
    
    print(f"Saved updated articles with 'top_k' field to {output_path}")


if __name__ == "__main__":
    main()
