"""Fine-tune a base model on the Uzbek embedding pairs.

CachedMultipleNegativesRankingLoss with in-batch negatives, one pass over the
(optionally subsampled) train split. The cached (GradCache) variant runs the
forward/backward in chunks of ``config.MINI_BATCH_SIZE`` so the full
``config.BATCH_SIZE`` worth of negatives fits on a free-tier T4. The model family
is selected with ``--group``; its base checkpoint, output repo and input prefixes
come from ``config.MODELS``.

    python -m src.train --group minilm
    python -m src.train --group e5_small
    python -m src.train --group minilm --smoke --no-push   # fast CPU dry run

On success the fine-tuned model is pushed to its Hub repo (unless ``--no-push`` or
``--smoke``), and always saved locally under ``outputs/``.
"""
import argparse

import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import config
from src import data


def _training_args(group: str, smoke: bool) -> SentenceTransformerTrainingArguments:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir(group, smoke)),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=8 if smoke else config.BATCH_SIZE,
        learning_rate=config.LR,
        warmup_ratio=config.WARMUP_RATIO,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        seed=config.SEED,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=not smoke,
    )


def train(group: str, smoke: bool = False, push: bool = True) -> SentenceTransformer:
    spec = config.MODELS[group]
    model = SentenceTransformer(spec["baseline"])
    model.max_seq_length = config.MAX_SEQ_LEN

    train_ds = data.load_train_dataset(group, smoke=smoke)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=config.MINI_BATCH_SIZE)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=_training_args(group, smoke),
        train_dataset=train_ds,
        loss=loss,
    )
    trainer.train()

    out_dir = str(config.output_dir(group, smoke))
    model.save(out_dir)
    if push and not smoke:
        try:
            model.push_to_hub(spec["finetuned"])
            print(f"Pushed to https://huggingface.co/{spec['finetuned']}")
        except Exception as exc:  # auth/network: the local save is the source of truth
            print(
                f"WARNING: push to the Hub failed -- the model is saved locally at {out_dir}.\n"
                f"  To push, run `huggingface-cli login` with a write token and re-run.\n"
                f"  underlying error: {exc}"
            )
    else:
        print(f"Saved locally to {out_dir} (push skipped)")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, choices=list(config.MODELS))
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU dry run on smoke_100")
    parser.add_argument("--no-push", action="store_true", help="Do not push to the Hub")
    args = parser.parse_args()
    train(args.group, smoke=args.smoke, push=not args.no_push)


if __name__ == "__main__":
    main()
