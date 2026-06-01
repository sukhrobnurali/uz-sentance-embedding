"""Build the held-out evaluation sets.

Two complementary tasks, each fed by an off-the-shelf sentence-transformers evaluator:

* Monolingual Uzbek retrieval -- title (query) against Wikipedia paragraph (passage),
  from the dataset's held-out ``wiki_retrieval_eval`` split.
* Cross-lingual uz<->en bitext -- FLORES+ ``devtest``, aligned by sentence id. Training
  only ever sees FLORES ``dev`` (via the dataset's validation split), so ``devtest``
  stays a clean held-out set.

Input prefixes are model-family specific (e5 needs them, MiniLM does not), so callers
pass the prefixes explicitly to keep each model's comparison like for like with training.
"""
from datasets import load_dataset

import config


def build_ir_eval(query_prefix: str, passage_prefix: str):
    """Return (queries, corpus, relevant_docs) for InformationRetrievalEvaluator."""
    ds = load_dataset(config.DATASET, config.IR_EVAL_CONFIG, split=config.IR_EVAL_SPLIT)

    queries, corpus, relevant_docs = {}, {}, {}
    for i, row in enumerate(ds):
        qid, did = f"q{i}", f"d{i}"
        queries[qid] = query_prefix + row["anchor"]
        corpus[did] = passage_prefix + row["positive"]
        relevant_docs[qid] = {did}
    return queries, corpus, relevant_docs


def build_flores_bitext(prefix: str):
    """Return (uz_sentences, en_sentences) parallel lists from FLORES+ devtest.

    Symmetric retrieval, so both sides get the same ``prefix`` (the family's query
    prefix per e5 guidance; empty for MiniLM). FLORES+ is gated -- the caller must be
    authenticated (``huggingface-cli login`` or HF_TOKEN) with the terms accepted.
    """
    uz = load_dataset(config.FLORES, config.FLORES_UZ, split=config.FLORES_SPLIT)
    en = load_dataset(config.FLORES, config.FLORES_EN, split=config.FLORES_SPLIT)

    en_by_id = {row["id"]: row["text"] for row in en}
    src, tgt = [], []
    for row in uz:
        match = en_by_id.get(row["id"])
        if match is not None:
            src.append(prefix + row["text"])
            tgt.append(prefix + match)
    return src, tgt
