from pathlib import Path
import os

root = Path(__file__).parent

logger_root = root.joinpath("logger")
dump_root = root.joinpath("dump")

hf_datasets_root = root.joinpath("datasets")

MODEL_ROOT = "/data/models"

model_to_ckpt = {
    "pythia": {
        "70m-deduped": f"{MODEL_ROOT}/pythia-70m-deduped",
        "160m-deduped": f"{MODEL_ROOT}/pythia-160m-deduped",
        "410m-deduped": f"{MODEL_ROOT}/pythia-410m-deduped",
        "1b-deduped": f"{MODEL_ROOT}/pythia-1b-deduped",
        "1.4b-deduped": f"{MODEL_ROOT}/pythia-1.4b-deduped",
        "2.8b-deduped": f"{MODEL_ROOT}/pythia-2.8b-deduped",
        "6.9b-deduped": f"{MODEL_ROOT}/pythia-6.9b-deduped",
        "12b-deduped": f"{MODEL_ROOT}/pythia-12b-deduped",
    },
    "opt": {
        "125m": f"{MODEL_ROOT}/opt-125m",
    },
    "bloom": {
        "560m": f"{MODEL_ROOT}/bloom-560m",
        "1.1b": f"{MODEL_ROOT}/bloom-1.1b",
    },
    "llama2": {
        "7b": f"{MODEL_ROOT}/Llama-2-7b-hf",
        "13b": f"{MODEL_ROOT}/Llama-2-13b-hf",
        "70b": f"{MODEL_ROOT}/Llama-2-70b-hf",
    },
    "gpt-neo": {
        "2.7b": f"{MODEL_ROOT}/gpt-neo-2.7B",
    },
    "gpt2": {
        "small": f"{MODEL_ROOT}/gpt2-small",
        "medium": f"{MODEL_ROOT}/gpt2-medium",
        "large": f"{MODEL_ROOT}/gpt2-large",
        "xl": f"{MODEL_ROOT}/gpt2-xl",
    },
    "fairseq": {
        "355m": f"{MODEL_ROOT}/Fairseq-dense-355M",
    },
}
