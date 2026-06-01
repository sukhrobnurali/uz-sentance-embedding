"""Load and prepare the training pairs for MultipleNegativesRankingLoss.

The dataset stores (anchor, positive) text pairs plus provenance columns. The
loss only needs the two text columns; the input prefixes a base model expects
(anchor -> query prefix, positive -> passage prefix) are applied here and depend
on the model family.
"""
from datasets import Dataset, concatenate_datasets, load_dataset

import config


def _make_prefixer(query_prefix: str, passage_prefix: str):
    def _add(batch):
        return {
            "anchor": [query_prefix + a for a in batch["anchor"]],
            "positive": [passage_prefix + p for p in batch["positive"]],
        }

    return _add


def _stratified_subsample(ds: Dataset, fraction: float) -> Dataset:
    """Keep ``fraction`` of each source so the 56/34/10 mix is preserved."""
    parts = []
    for src in sorted(set(ds["source"])):
        sub = ds.filter(lambda r: r["source"] == src).shuffle(seed=config.SEED)
        n = max(1, int(len(sub) * fraction))
        parts.append(sub.select(range(n)))
    return concatenate_datasets(parts).shuffle(seed=config.SEED)


def load_train_dataset(group: str, smoke: bool = False) -> Dataset:
    """Return a Dataset with exactly the ``anchor``/``positive`` columns, prefixed
    for the given model family (``group`` key into ``config.MODELS``)."""
    spec = config.MODELS[group]
    if smoke:
        ds = load_dataset(config.DATASET, config.SMOKE_CONFIG, split="train")
    else:
        ds = load_dataset(config.DATASET, config.TRAIN_CONFIG, split=config.TRAIN_SPLIT)
        if config.TRAIN_FRACTION < 1.0:
            ds = _stratified_subsample(ds, config.TRAIN_FRACTION)

    prefixer = _make_prefixer(spec["query_prefix"], spec["passage_prefix"])
    ds = ds.map(prefixer, batched=True)
    drop = [c for c in ds.column_names if c not in ("anchor", "positive")]
    return ds.remove_columns(drop)
