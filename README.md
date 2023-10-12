# uj2vec

Application of Transformer neural networks for representation learning on user journeys (product telemetry event sequences). Makes use of TransformerXL for learning representations that span multiple segments to accomodate very long user journeys. Contrastive loss function coupled with the transformer's memory allows the model to learn representations that are invariant to which segment is in view.

## Table of Contents

- Prerequisites
- Usage

## Prerequisites

TransformerXL - https://github.com/kimiyoung/transformer-xl/tree/master/pytorch

## Usage

### Initialize the model:

`model = UJ2Vec(dropout=0.1, ntoken=26, bptt=100, embed_dim = 64, z_dim=256, \
 hidden_dim=512, nhead=4, nlayers=2)

### Encode a batch of event sequence tokens with shape according to length of the sequence and batch size
```
# An S-length segment of the sequence
event_tokens = torch.rand(100, 10).int() # [S, B]

# Select last token as sequence representation
z = model.encoder(event_tokens)[..., -1:, :, :] # [1, B, Z]
```

### Project into joint space between current segment and random previous segment view
```
u = model.decoder[0](z)
v = model.decoder[1](previous_z)
```

### Compute the loss based on cosine similarity between current view and a previous view.

`loss = model.loss(u, v)

### Reset encoder memory
```
model.encoder.mems = None
```

