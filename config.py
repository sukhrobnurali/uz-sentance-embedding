"""Central configuration.

All model ids, dataset coordinates, hyperparameters and paths live here so that
nothing is hard-coded inside the logic modules.

Two base families are studied. e5 requires ``query:``/``passage:`` input prefixes;
MiniLM uses none. Prefixes therefore live per-model in ``MODELS`` and are threaded
through training and evaluation so both models are treated correctly.
"""
from pathlib import Path

# --- Model families: base -> fine-tuned, with the prefixes each base expects ---
MODELS = {
    "e5_small": {
        "baseline": "intfloat/multilingual-e5-small",
        "finetuned": "sukhrobnurali/uzbek-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "minilm": {
        "baseline": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "finetuned": "sukhrobnurali/uzbek-minilm",
        "query_prefix": "",
        "passage_prefix": "",
    },
}

# --- Training data (sukhrobnurali/uzbek-embedding-pairs) ---
DATASET = "sukhrobnurali/uzbek-embedding-pairs"
TRAIN_CONFIG = "default"
TRAIN_SPLIT = "train"
SMOKE_CONFIG = "smoke_100"
IR_EVAL_CONFIG = "wiki_retrieval_eval"
IR_EVAL_SPLIT = "test"

# --- FLORES+ cross-lingual eval (gated: needs HF auth + accepted terms) ---
FLORES = "openlanguagedata/flores_plus"
FLORES_UZ = "uzn_Latn"
FLORES_EN = "eng_Latn"
FLORES_SPLIT = "devtest"

# --- Training hyperparameters ---
SEED = 42
EPOCHS = 1
BATCH_SIZE = 192            # in-batch negatives; fits an A100 40GB with margin (drop to ~48 on a free T4)
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 192
TRAIN_FRACTION = 1.0        # <1.0 stratified-subsamples the train split by source

# --- Evaluation ---
K_VALUES = [1, 5, 10]

# --- Paths ---
ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "metrics.json"


def output_dir(group: str, smoke: bool = False) -> Path:
    """Local save path for a fine-tuned model, shared by training and evaluation."""
    name = MODELS[group]["finetuned"].split("/")[-1]
    return ROOT / "outputs" / (f"{name}-smoke" if smoke else name)
