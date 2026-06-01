---
language:
- uz
- en
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
datasets:
- sukhrobnurali/uzbek-embedding-pairs
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- embeddings
- uzbek
- retrieval
- minilm
model-index:
- name: uzbek-minilm
  results:
  - task:
      type: information-retrieval
      name: Monolingual Uzbek retrieval (Wikipedia title to paragraph)
    dataset:
      type: sukhrobnurali/uzbek-embedding-pairs
      name: uzbek-embedding-pairs (wiki_retrieval_eval/test, 5k held-out)
    metrics:
    - type: recall_at_1
      value: 0.9692
      name: Recall@1
    - type: recall_at_5
      value: 0.9854
      name: Recall@5
    - type: recall_at_10
      value: 0.9892
      name: Recall@10
    - type: mrr_at_10
      value: 0.9765
      name: MRR@10
    - type: ndcg_at_10
      value: 0.9796
      name: nDCG@10
  - task:
      type: bitext-mining
      name: Cross-lingual uz-en bitext retrieval (FLORES+ devtest)
    dataset:
      type: openlanguagedata/flores_plus
      name: FLORES+ devtest (uzn_Latn / eng_Latn, 1012 pairs)
    metrics:
    - type: accuracy
      value: 0.8498
      name: Mean accuracy (both directions)
---

# uzbek-minilm

A multilingual sentence-embedding model fine-tuned from
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
for **Uzbek semantic search and retrieval**, including cross-lingual uz↔en.

This model demonstrates the large gain achievable when the base model is weak at the
target language. The base MiniLM handles Uzbek poorly (Recall@1 = 0.26); after one epoch
on Uzbek pairs it reaches Recall@1 = 0.97. The companion flagship
[`sukhrobnurali/uzbek-e5-small`](https://huggingface.co/sukhrobnurali/uzbek-e5-small)
starts from a stronger base and is the recommended model for cross-lingual work.

- **Base model:** `paraphrase-multilingual-MiniLM-L12-v2` (118M params, 384-dim, Apache 2.0)
- **Language:** Uzbek (Latin), with English for cross-lingual pairs
- **Training data:** [`sukhrobnurali/uzbek-embedding-pairs`](https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs)
- **Objective:** `MultipleNegativesRankingLoss`, 1 epoch
- **Training code:** https://github.com/sukhrobnurali/uz-sentance-embedding

## Intended use

- Uzbek semantic search / passage retrieval (RAG over Uzbek documents).
- Cross-lingual uz↔en retrieval and bitext alignment.
- General-purpose Uzbek sentence similarity and clustering.

Unlike e5, this model uses **no input prefixes** — encode queries and documents as plain
text. Embeddings are L2-normalized; compare with cosine similarity.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sukhrobnurali/uzbek-minilm")

query = ["O'zbekistonning poytaxti qaysi shahar?"]
passages = [
    "Toshkent — O'zbekiston Respublikasining poytaxti va eng yirik shahri.",
    "Samarqand — O'zbekistondagi qadimiy shaharlardan biri.",
]

q_emb = model.encode(query, normalize_embeddings=True)
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
| Recall@1 | 0.2564 | **0.9692** | +0.7128 |
| Recall@5 | 0.3742 | **0.9854** | +0.6112 |
| Recall@10 | 0.4314 | **0.9892** | +0.5578 |
| MRR@10 | 0.3072 | **0.9765** | +0.6693 |
| nDCG@10 | 0.3367 | **0.9796** | +0.6429 |

### Cross-lingual uz↔en bitext — FLORES+ `devtest` (1,012 pairs)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| uz→en accuracy | 0.4575 | **0.8745** | +0.4170 |
| en→uz accuracy | 0.4872 | **0.8251** | +0.3379 |
| Mean accuracy | 0.4723 | **0.8498** | +0.3775 |

Fine-tuning nearly closes the monolingual gap — the fine-tuned MiniLM (R@1 = 0.969) almost
matches the e5 *baseline* (0.987). On cross-lingual FLORES it still trails the e5 family
(0.85 vs 0.99), so prefer `uzbek-e5-small` when uz↔en accuracy matters most.

## Training data

[`sukhrobnurali/uzbek-embedding-pairs`](https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs)
— 356,278 `(anchor, positive)` pairs in the `default/train` split, used as-is:

| Source | Share | Pair type |
|---|---|---|
| Uzbek Wikipedia | ~56% | title ↔ paragraph |
| OPUS-100 uz↔en | ~34% | parallel sentence ↔ translation |
| Latin↔Cyrillic | ~10% | same sentence, two scripts |

No prefixes are applied for this model family.

## Limitations

- Low-resource language: coverage is thinner than for high-resource languages, and rare
  domains (legal, medical, dialectal) are under-represented.
- Part of the data is OPUS-100, which carries machine-translation noise.
- The monolingual eval is a title→paragraph proxy for retrieval, not a curated query set.
- Trained on Latin-script Uzbek; Cyrillic appears only via the script-pair source, so
  Cyrillic retrieval quality is less thoroughly evaluated.
- Cross-lingual uz↔en accuracy trails the e5-based model; use `uzbek-e5-small` for that.

## Reproducibility

Fixed seed (42); 1 epoch of `MultipleNegativesRankingLoss` with in-batch negatives,
batch size 192, lr 2e-5, 10% warmup, `max_seq_length=192`, bf16 on Ampere. All
hyperparameters live in `config.py`; training and evaluation scripts are in the
[training repository](https://github.com/sukhrobnurali/uz-sentance-embedding).

## License

Apache 2.0, inherited from the base model `paraphrase-multilingual-MiniLM-L12-v2`.
