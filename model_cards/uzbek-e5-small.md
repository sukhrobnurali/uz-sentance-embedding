---
language:
- uz
- en
license: mit
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: intfloat/multilingual-e5-small
datasets:
- sukhrobnurali/uzbek-embedding-pairs
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- embeddings
- uzbek
- retrieval
- e5
model-index:
- name: uzbek-e5-small
  results:
  - task:
      type: information-retrieval
      name: Monolingual Uzbek retrieval (Wikipedia title to paragraph)
    dataset:
      type: sukhrobnurali/uzbek-embedding-pairs
      name: uzbek-embedding-pairs (wiki_retrieval_eval/test, 5k held-out)
    metrics:
    - type: recall_at_1
      value: 0.9914
      name: Recall@1
    - type: recall_at_5
      value: 0.9964
      name: Recall@5
    - type: recall_at_10
      value: 0.9972
      name: Recall@10
    - type: mrr_at_10
      value: 0.9936
      name: MRR@10
    - type: ndcg_at_10
      value: 0.9945
      name: nDCG@10
  - task:
      type: bitext-mining
      name: Cross-lingual uz-en bitext retrieval (FLORES+ devtest)
    dataset:
      type: openlanguagedata/flores_plus
      name: FLORES+ devtest (uzn_Latn / eng_Latn, 1012 pairs)
    metrics:
    - type: accuracy
      value: 0.9876
      name: Mean accuracy (both directions)
---

# uzbek-e5-small

A small multilingual sentence-embedding model fine-tuned from
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
for **Uzbek semantic search and retrieval**, including cross-lingual uz↔en.

This is the flagship model of a two-base study. Its base, `multilingual-e5-small`, is
already strong at Uzbek, so fine-tuning yields a small but consistently positive gain —
the honest result for a near-ceiling starting point. The companion
[`sukhrobnurali/uzbek-minilm`](https://huggingface.co/sukhrobnurali/uzbek-minilm) starts
from a much weaker base and shows the large delta.

- **Base model:** `intfloat/multilingual-e5-small` (118M params, 384-dim, MIT)
- **Language:** Uzbek (Latin), with English for cross-lingual pairs
- **Training data:** [`sukhrobnurali/uzbek-embedding-pairs`](https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs)
- **Objective:** `MultipleNegativesRankingLoss`, 1 epoch
- **Training code:** https://github.com/sukhrobnurali/uz-sentance-embedding

## Intended use

- Uzbek semantic search / passage retrieval (RAG over Uzbek documents).
- Cross-lingual uz↔en retrieval and bitext alignment.
- General-purpose Uzbek sentence similarity and clustering.

### Prefixes (important)

Like the base e5 model, this model expects task prefixes:

- Retrieval: prefix queries with `query: ` and documents with `passage: `.
- Symmetric tasks (similarity, bitext): use `query: ` on both sides.

Embeddings are L2-normalized; compare with cosine similarity (dot product on normalized
vectors).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sukhrobnurali/uzbek-e5-small")

queries = ["query: O'zbekistonning poytaxti qaysi shahar?"]
passages = [
    "passage: Toshkent — O'zbekiston Respublikasining poytaxti va eng yirik shahri.",
    "passage: Samarqand — O'zbekistondagi qadimiy shaharlardan biri.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
p_emb = model.encode(passages, normalize_embeddings=True)
scores = q_emb @ p_emb.T
print(scores)  # highest score on the Tashkent passage
```

> Requires `sentence-transformers>=5.5.1` — the version the model was saved with.
> Older versions cannot load it (`ModuleNotFoundError: No module named 'sentence_transformers.base'`).

## Evaluation

The same protocol is applied to the base and fine-tuned models so the delta is a fair
comparison. Both held-out sets are disjoint from training: the retrieval split is the
dataset's `wiki_retrieval_eval/test`; FLORES+ training only ever sees `dev` (via the
dataset's validation split), so `devtest` stays clean.

### Monolingual Uzbek retrieval — `wiki_retrieval_eval/test` (5,000 title→paragraph)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Recall@1 | 0.9868 | **0.9914** | +0.0046 |
| Recall@5 | 0.9948 | **0.9964** | +0.0016 |
| Recall@10 | 0.9962 | **0.9972** | +0.0010 |
| MRR@10 | 0.9904 | **0.9936** | +0.0032 |
| nDCG@10 | 0.9918 | **0.9945** | +0.0027 |

### Cross-lingual uz↔en bitext — FLORES+ `devtest` (1,012 pairs)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| uz→en accuracy | 0.9713 | **0.9901** | +0.0188 |
| en→uz accuracy | 0.9852 | 0.9852 | +0.0000 |
| Mean accuracy | 0.9783 | **0.9876** | +0.0094 |

The base is already near the ceiling for Uzbek, so the gains are small but every metric
improves — the fine-tuned model dominates the base and is the one shipped.

## Training data

[`sukhrobnurali/uzbek-embedding-pairs`](https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs)
— 356,278 `(anchor, positive)` pairs in the `default/train` split, used as-is:

| Source | Share | Pair type |
|---|---|---|
| Uzbek Wikipedia | ~56% | title ↔ paragraph |
| OPUS-100 uz↔en | ~34% | parallel sentence ↔ translation |
| Latin↔Cyrillic | ~10% | same sentence, two scripts |

Anchors are prefixed `query: ` and positives `passage: ` at training time, matching the
retrieval framing used at inference.

## Limitations

- Low-resource language: coverage is thinner than for high-resource languages, and rare
  domains (legal, medical, dialectal) are under-represented.
- Part of the data is OPUS-100, which carries machine-translation noise.
- The monolingual eval is a title→paragraph proxy for retrieval, not a curated query set.
- Trained on Latin-script Uzbek; Cyrillic appears only via the script-pair source, so
  Cyrillic retrieval quality is less thoroughly evaluated.

## Reproducibility

Fixed seed (42); 1 epoch of `MultipleNegativesRankingLoss` with in-batch negatives,
batch size 192, lr 2e-5, 10% warmup, `max_seq_length=192`, bf16 on Ampere. All
hyperparameters live in `config.py`; training and evaluation scripts are in the
[training repository](https://github.com/sukhrobnurali/uz-sentance-embedding).

## License

MIT, inherited from the base model `intfloat/multilingual-e5-small`.
