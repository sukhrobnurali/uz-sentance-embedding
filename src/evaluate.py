"""Evaluate a sentence-embedding model on the held-out Uzbek tasks.

Runs two evaluations and merges the numbers into ``results/metrics.json``:
  * monolingual Uzbek Wikipedia retrieval (Recall@k, MRR, nDCG)
  * cross-lingual uz<->en bitext accuracy on FLORES+ devtest

Results are keyed by model family (``--group``) and stage (``--stage``); run each
family's base and fine-tuned model so the script can compute the per-family delta.

    python -m src.evaluate --group e5_small --stage baseline
    python -m src.evaluate --group e5_small --stage finetuned
    python -m src.evaluate --group minilm   --stage baseline

The model id and the input prefixes are looked up from ``config.MODELS``. Pass
``--model`` to override the id (e.g. a local checkpoint path).

FLORES+ is gated. If you have not accepted its terms / set a token, pass
``--skip-flores`` to run the monolingual task only (recorded explicitly, never silently).
"""
import argparse
import json

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    TranslationEvaluator,
)

import config
from src import eval_sets

IR_NAME = "uz_wiki"
FLORES_NAME = "flores_uz_en"


def _run_ir(model, query_prefix: str, passage_prefix: str) -> dict:
    queries, corpus, relevant_docs = eval_sets.build_ir_eval(query_prefix, passage_prefix)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        accuracy_at_k=config.K_VALUES,
        precision_recall_at_k=config.K_VALUES,
        mrr_at_k=config.K_VALUES,
        ndcg_at_k=config.K_VALUES,
        name=IR_NAME,
        show_progress_bar=True,
    )
    return dict(evaluator(model))


def _run_flores(model, prefix: str) -> dict:
    src, tgt = eval_sets.build_flores_bitext(prefix)
    evaluator = TranslationEvaluator(
        source_sentences=src,
        target_sentences=tgt,
        name=FLORES_NAME,
        show_progress_bar=True,
    )
    return dict(evaluator(model))


def _load_path(group: str, stage: str, override: str | None) -> str:
    """Where to load weights from. The finetuned stage prefers the local checkpoint
    written by training, so evaluation never depends on a successful Hub push; it
    falls back to the Hub id only if no local checkpoint exists."""
    if override:
        return override
    if stage == "finetuned":
        local = config.output_dir(group)
        if local.exists():
            print(f"[eval] loading local checkpoint: {local}")
            return str(local)
        print(f"[eval] no local checkpoint at {local}; loading {config.MODELS[group][stage]} from the Hub")
    return config.MODELS[group][stage]


def _load_results() -> dict:
    if config.RESULTS_PATH.exists():
        return json.loads(config.RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_results(results: dict) -> None:
    config.RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _delta(base: dict, tuned: dict) -> dict:
    """Per-task delta (finetuned - baseline) over shared numeric metrics."""
    out = {}
    for task in (IR_NAME, FLORES_NAME):
        b, t = base.get(task, {}), tuned.get(task, {})
        if isinstance(b, dict) and isinstance(t, dict):
            shared = {
                k: round(t[k] - v, 4)
                for k, v in b.items()
                if isinstance(v, (int, float)) and isinstance(t.get(k), (int, float))
            }
            if shared:
                out[task] = shared
    return out


def _compute_all_deltas(results: dict) -> dict:
    deltas = {}
    for group in config.MODELS:
        stages = results.get(group, {})
        if "baseline" in stages and "finetuned" in stages:
            deltas[group] = _delta(stages["baseline"], stages["finetuned"])
    return deltas


def _print_summary(results: dict) -> None:
    print("\n==================== metrics ====================")
    for group in config.MODELS:
        stages = results.get(group, {})
        for stage in ("baseline", "finetuned"):
            entry = stages.get(stage)
            if not entry:
                continue
            print(f"\n[{group}/{stage}] {entry.get('model')}")
            for task in (IR_NAME, FLORES_NAME):
                scores = entry.get(task, {})
                if scores.get("status") in ("skipped", "failed"):
                    print(f"  {task}: {scores['status']} ({scores.get('reason')})")
                    continue
                for k, v in scores.items():
                    if isinstance(v, (int, float)):
                        print(f"  {task} | {k}: {v:.4f}")
    for group, tasks in results.get("deltas", {}).items():
        print(f"\n[delta {group}] finetuned - baseline")
        for task, scores in tasks.items():
            for k, v in scores.items():
                print(f"  {task} | {k}: {v:+.4f}")
    print("=================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--group", required=True, choices=list(config.MODELS))
    parser.add_argument("--stage", required=True, choices=["baseline", "finetuned"])
    parser.add_argument("--model", help="Override the model id (e.g. a local checkpoint)")
    parser.add_argument(
        "--skip-flores",
        action="store_true",
        help="Run monolingual IR only; record FLORES as explicitly skipped",
    )
    args = parser.parse_args()

    spec = config.MODELS[args.group]
    model_id = args.model or spec[args.stage]  # canonical id recorded in results
    model = SentenceTransformer(_load_path(args.group, args.stage, args.model))
    model.max_seq_length = config.MAX_SEQ_LEN

    entry = {
        "model": model_id,
        IR_NAME: _run_ir(model, spec["query_prefix"], spec["passage_prefix"]),
    }

    if args.skip_flores:
        entry[FLORES_NAME] = {"status": "skipped", "reason": "--skip-flores"}
        print("\n[NOTE] FLORES cross-lingual eval skipped (--skip-flores).")
    else:
        try:
            entry[FLORES_NAME] = _run_flores(model, spec["query_prefix"])
        except Exception as exc:  # gated dataset / auth / network -- keep the IR results
            entry[FLORES_NAME] = {"status": "failed", "reason": str(exc)}
            print(
                "\n[WARN] FLORES+ eval failed -- IR results kept, FLORES recorded as failed.\n"
                "  It is gated: accept the terms at "
                "https://huggingface.co/datasets/openlanguagedata/flores_plus and authenticate "
                "(HF_TOKEN / huggingface-cli login), or pass --skip-flores to suppress this.\n"
                f"  underlying error: {exc}"
            )

    results = _load_results()
    results.setdefault(args.group, {})[args.stage] = entry
    results["deltas"] = _compute_all_deltas(results)
    _save_results(results)
    _print_summary(results)


if __name__ == "__main__":
    main()
