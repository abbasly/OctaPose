# src/train.py

import torch.nn as nn
from tqdm import tqdm

def train_model(model, dataloader, optimizer, alpha=0.3, device="cpu", epoch = 20):
    model.to(device)
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=0.3)

    for epoch in range(epoch):
        total = 0
        correct = 0
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            emb_a, logits = model(anchor, lengths)
            emb_p, _ = model(positive, lengths)
            emb_n, _ = model(negative, lengths)

            loss_cls = ce_loss(logits, labels)
            loss_trip = triplet_loss(emb_a, emb_p, emb_n)
            loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        # print(f"Epoch {epoch+1} | Triplet Loss: {loss_trip:.4f} | Cls Loss: {loss_cls:.4f}")
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.2%}")
