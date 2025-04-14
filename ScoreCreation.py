import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GTE model + tokenizer
model_name = "thenlper/gte-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()


# Step 1: Preprocess
def preprocess_paragraphs(df, section_col='sections', output_col='paragraphs'):
    def flatten(sections):
        return [p.strip() for section in sections for p in section if isinstance(p, str)]
    df[output_col] = df[section_col].apply(flatten)
    return df


# Step 2: Embedding using mean pooling
def mean_pooling(last_hidden_state, attention_mask):
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
    sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_embeddings(paragraphs, summary):
    all_texts = paragraphs + [summary]
    encoded_input = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
    para_embs = embeddings[:-1].cpu().numpy()
    summary_emb = embeddings[-1].cpu().numpy()
    return para_embs, summary_emb


# Step 3: Similarity
def compute_similarity_scores(para_embs, summary_emb):
    sim_scores = cosine_similarity(para_embs, summary_emb.reshape(1, -1)).flatten()
    return sim_scores


# Step 4: Full processing per row
def compute_para_similarities(df, summary_col='summary', section_col='sections'):
    df = preprocess_paragraphs(df, section_col=section_col, output_col='paragraphs')

    all_scores = []
    for count, row in enumerate(df.itertuples(index=False), 1):
        paragraphs = row.paragraphs
        summary_list = getattr(row, summary_col)
        summary = " ".join(s.strip() for s in summary_list if isinstance(s, str))

        if not paragraphs or not summary:
            all_scores.append([])
            continue
        # print('Data Prepared')
        para_embs, summary_emb = compute_embeddings(paragraphs, summary)
        # print('Data Embedded')
        scores = compute_similarity_scores(para_embs, summary_emb)
        # print('Score Calculated')
        # print(scores.tolist())
        all_scores.append(scores.tolist())

        if count % 10 == 0:
            print(f"[GPU] Processed {count} rows")

    df['similarity_scores'] = all_scores
    return df


if __name__ == '__main__':
    core = pd.read_json('../elife/test.json')
    print(f"Loaded {core.shape[0]} examples.")
    out_df = compute_para_similarities(core, summary_col='summary', section_col='sections')
    out_df.to_csv('../elife/test_simscore.csv', index=False)
