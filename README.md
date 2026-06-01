# uzbek-e5-small — Uzbek sentence-embedding fine-tune

Fine-tunes [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
for Uzbek semantic search / retrieval, with an honest baseline-vs-fine-tuned evaluation
on a held-out set.

- **Base model:** `intfloat/multilingual-e5-small` (118M, 384-dim, MIT)
- **Training data:** [`sukhrobnurali/uzbek-embedding-pairs`](https://huggingface.co/datasets/sukhrobnurali/uzbek-embedding-pairs)
  (356k `(anchor, positive)` pairs: Uzbek Wikipedia, OPUS-100 uz↔en, Latin↔Cyrillic)
- **Objective:** `MultipleNegativesRankingLoss`, 1 epoch — batch 192 of in-batch negatives (tuned for an A100 40GB; lower for a free T4)
- **Output model:** `sukhrobnurali/uzbek-e5-small`

## Layout

| Path | Purpose |
|---|---|
| `config.py` | All model ids, dataset coordinates, hyperparameters, paths |
| `src/data.py` | Load the train split, apply e5 prefixes, optional stratified subsample |
| `src/eval_sets.py` | Build the monolingual IR set and the FLORES+ bitext set |
| `src/train.py` | Train (MNRL) and push to the Hub |
| `src/evaluate.py` | Baseline vs fine-tuned metrics → `results/metrics.json` |
| `notebooks/train_colab.ipynb` | Thin Colab orchestrator |

## Evaluation

Same protocol applied to the base and fine-tuned models so the delta is fair:

- **Monolingual Uzbek retrieval** — `wiki_retrieval_eval` held-out split (5k title→paragraph);
  Recall@1/5/10, MRR@10, nDCG@10.
- **Cross-lingual uz↔en bitext** — FLORES+ `devtest` (1,012; never trained on); accuracy both
  directions. FLORES+ is gated — accept its terms and `huggingface-cli login`, or pass
  `--skip-flores` to run the monolingual task only.

```bash
python -m src.evaluate --model intfloat/multilingual-e5-small --name baseline
python -m src.evaluate --model sukhrobnurali/uzbek-e5-small  --name finetuned
```

## Reproducibility

Fixed seed (`config.SEED = 42`); all hyperparameters in `config.py`. See `requirements.txt`.
