import torch
import torch.nn as nn

class KOEmbeddingModel(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=256, embedding_dim=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # x: [batch_size, 150, 34]
        _, h_n = self.gru(x)  # h_n: [num_layers, batch_size, hidden_dim]
        last_hidden = h_n[-1]  # Take final hidden state from last GRU layer
        embedding = self.embedding_head(last_hidden)  # [batch_size, embedding_dim]
        return embedding
