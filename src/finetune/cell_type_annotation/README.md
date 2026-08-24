# Zero-shot cell-type annotation

This package implements scGPT-style reference mapping with frozen `[CLS]`
embeddings and ten-nearest-neighbour majority voting. Dataset preparation,
paths, model states, and submission instructions are documented in
[`../batch_integration/README.md`](../batch_integration/README.md) because both
single-cell tasks deliberately use the same cells and selected genes.
