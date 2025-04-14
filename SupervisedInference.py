import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class RobertaScoreRegressor(nn.Module):
    def __init__(self, pretrained_model='roberta-base'):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model)
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return self.regressor(cls_embedding).squeeze(-1)


class InferenceParaDataset(Dataset):
    def __init__(self, paragraphs, tokenizer, max_length=256):
        self.paragraphs = paragraphs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.paragraphs[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }


def explode_paragraphs(df):
    if df['paragraphs'].apply(type).eq(str).any():
        df['paragraphs'] = df['paragraphs'].apply(ast.literal_eval)

    df = df.explode('paragraphs', ignore_index=True)
    df = df.rename(columns={'paragraphs': 'paragraph'})
    return df


def predict_scores(paragraphs, model, tokenizer, batch_size=16):
    model.eval()
    dataset = InferenceParaDataset(paragraphs, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            preds = model(input_ids, attention_mask)
            all_scores.extend(preds.cpu().tolist())

    return all_scores


if __name__ == '__main__':
    print("Loading paragraphs for inference...")
    df = pd.read_json('../plos/test_simscore.json')
    df = explode_paragraphs(df)
    paragraphs = df['paragraph'].tolist()
    print(f"Loaded {len(paragraphs)} paragraphs.")

    # Load model + tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaScoreRegressor().to(device)
    model.load_state_dict(torch.load("roberta_score_regressor.pt", map_location=device))

    print("Predicting scores...")
    predicted_scores = predict_scores(paragraphs, model, tokenizer)

    # Save predictions
    df['predicted_score'] = predicted_scores
    df.to_csv('../plos/test_simscore_predictions.csv', index=False)
    print("Saved predictions to test_predictions.csv")
