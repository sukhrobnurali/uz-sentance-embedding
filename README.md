# Uzbek sentence-embedding fine-tunes

Fine-tunes small multilingual embedding models for **Uzbek** semantic search / retrieval
(monolingual and cross-lingual uz↔en), with an honest baseline-vs-fine-tuned evaluation on
held-out data.

This is a **two-base study**: the same pipeline, data, and eval are applied to two bases
with very different Uzbek ability, which makes the fine-tuning gain interpretable rather
than a single number in isolation.

| Model | Base | Hub | Role |
|---|---|---|---|
| `uzbek-e5-small` | `intfloat/multilingual-e5-small` | [sukhrobnurali/uzbek-e5-small](https://huggingface.co/sukhrobnurali/uzbek-e5-small) | Flagship — strong base, marginal but all-positive gain |
| `uzbek-minilm` | `paraphrase-multilingual-MiniLM-L12-v2` | [sukhrobnurali/uzbek-minilm](https://huggingface.co/sukhrobnurali/uzbek-minilm) | Demo — weak base, large gain |

## Results

Headline numbers (full tables and per-direction breakdowns in [`results/metrics.json`](results/metrics.json)
and the [model cards](model_cards/)). Recall@1 is monolingual Uzbek retrieval; FLORES is the
mean uz↔en bitext accuracy.

| Model | Recall@1 (base → ft) | FLORES mean (base → ft) |
|---|---|---|
| `uzbek-e5-small` | 0.987 → **0.991** (+0.005) | 0.978 → **0.988** (+0.009) |
| `uzbek-minilm` | 0.256 → **0.969** (+0.713) | 0.472 → **0.850** (+0.378) |

The takeaway: e5-small is already near the ceiling for Uzbek, so fine-tuning helps only a
little — but every metric improves, so the fine-tuned model still ships. MiniLM starts weak
and improves dramatically; fine-tuned MiniLM (0.969 R@1) nearly matches the e5 *baseline*
(0.987) on monolingual retrieval, while still trailing e5 on cross-lingual FLORES (0.85 vs
0.99). e5-small is the model to use when uz↔en accuracy matters.

## Layout

| Path | Purpose |
|---|---|
| `config.py` | All model ids, dataset coordinates, hyperparameters, paths, per-model prefixes |
| `src/data.py` | Load the train split, apply per-family prefixes, optional stratified subsample |
| `src/eval_sets.py` | Build the monolingual IR set and the FLORES+ bitext set |
| `src/train.py` | Train (MNRL) and push to the Hub (`--group e5_small` or `--group minilm`) |
| `src/evaluate.py` | Baseline vs fine-tuned metrics → `results/metrics.json` |
| `notebooks/train_colab.ipynb` | Thin Colab orchestrator (train → evaluate) |
| `model_cards/` | Per-model Hub cards (`uzbek-e5-small.md`, `uzbek-minilm.md`) |
| `results/metrics.json` | Baseline + fine-tuned numbers and deltas for both models |

Logic lives in versioned `.py` modules; the notebook only orchestrates on Colab. No
hyperparameters are hard-coded inside the logic modules — everything is in `config.py`.

## Data and prefixes

Training data is [`sukhrobnurali/uzbek-embedding-pairs`](https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs)
used as-is: 356k `(anchor, positive)` pairs (≈56% Uzbek Wikipedia title↔paragraph, ≈34%
OPUS-100 uz↔en, ≈10% Latin↔Cyrillic).

The two bases need different input conventions: **e5 requires `query:`/`passage:` prefixes,
MiniLM uses none.** Prefixes are declared per model in `config.MODELS` and threaded through
training and evaluation so each model is treated correctly and the base-vs-fine-tuned
comparison stays like-for-like.

## Evaluation

The same protocol is applied to the base and fine-tuned models so the delta is fair:

- **Monolingual Uzbek retrieval** — `wiki_retrieval_eval/test` held-out split (5k
  title→paragraph); Recall@1/5/10, MRR@10, nDCG@10.
- **Cross-lingual uz↔en bitext** — FLORES+ `devtest` (1,012; never trained on); accuracy
  both directions. FLORES+ is gated — accept its terms and `huggingface-cli login`, or pass
  `--skip-flores` to run the monolingual task only.

```bash
# e5 family
python -m src.evaluate --group e5_small --stage baseline
python -m src.evaluate --group e5_small --stage finetuned

# MiniLM family
python -m src.evaluate --group minilm --stage baseline
python -m src.evaluate --group minilm --stage finetuned
```

## Training

GPU-only (tuned for an A100 40GB: plain `MultipleNegativesRankingLoss`, batch 192, bf16).
On a free-tier T4, drop `BATCH_SIZE` to ~48.

```bash
python -m src.train --group minilm
python -m src.train --group e5_small
python -m src.train --group minilm --smoke --no-push   # fast CPU dry run
```

## Reproducibility

Fixed seed (`config.SEED = 42`); all hyperparameters in `config.py`. See `requirements.txt`.
