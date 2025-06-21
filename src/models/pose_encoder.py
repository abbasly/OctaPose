import torch.nn as nn
import torch
import torch.nn.functional as F


class PoseGRUEncoder(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=256, embed_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x, lengths):
        B, T, J, C = x.shape  # [B, T, 17, 2]
        x = x.view(B, T, J * C)  # flatten keypoints: [B, T, 34]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # h_n: [2, B, H]
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # [B, 2H]
        emb = F.normalize(self.fc(h), dim=1)  # [B, D]
        return emb
    
class PoseGRUHybrid(nn.Module):
    def __init__(self, num_classes, embed_dim=3):
        super().__init__()
        self.encoder = PoseGRUEncoder(embed_dim=embed_dim)  # same GRU block
        self.fc_cls = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, lengths):
        emb = self.encoder(x, lengths)                      # (B, D)
        logits = self.fc_cls(emb)                           # (B, C)
        return emb, logits