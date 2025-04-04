import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
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


class ParaScoreDataset(Dataset):
    def __init__(self, paragraphs, scores, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.paragraphs = paragraphs
        self.scores = scores
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
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'score': torch.tensor(self.scores[idx], dtype=torch.float)
        }


def flatten_paragraphs(df):
    rows = []
    for _, row in df.iterrows():
        try:
            paragraphs = ast.literal_eval(row['paragraphs'])
            scores = ast.literal_eval(row['similarity_scores'])
        except Exception as e:
            continue
        for p, s in zip(paragraphs, scores):
            rows.append({'paragraph': p, 'score': s})
    return pd.DataFrame(rows)


def train_model(train_df, epochs=5, batch_size=16):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaScoreRegressor().to(device)

    dataset = ParaScoreDataset(train_df['paragraph'].tolist(), train_df['score'].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['score'].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    return model


if __name__ == '__main__':
    print("Loading dataset...")
    df = pd.read_csv('../plos/train_simscore.csv')
    df = df.loc[:1000]
    print("Flattening paragraphs...")
    train_df = flatten_paragraphs(df)
    print(f"Training on {len(train_df)} paragraph-score pairs.")

    trained_model = train_model(train_df)
    torch.save(trained_model.state_dict(), "roberta_score_regressor.pt")
