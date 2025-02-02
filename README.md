# Iterative Forward Tuning Boosts In-Context Learning in Language Models

This repository is the official implementation of the paper [Iterative Forward Tuning Boosts In-Context Learning in Language Models](https://arxiv.org/abs/2305.13016).

## Quick Start

The main entry point is `main.py`. The CLI is currently in **Debug mode**, so command-line arguments are ignored and just run:

```bash
python main.py
```

This will load the model (OPT-125M by default), run Iterative Forward Tuning evaluation on the SST2 task, and output metrics such as `lm_log_p` during iterations. Logs and results are saved under `logger/DEBUG/` and `dump/DEBUG/` respectively.

## Project Structure

- `main.py` — Entry point; orchestrates model loading, task evaluation, and result dumping
- `core.py` — Core logic for generative and selective (classification) tasks
- `models/` — Model loading and Iterative Forward Tuning (KV iteration) implementation
- `tasks/` — Task definitions (SST2, ARC, BBH, MMLU, etc.)
- `anchor.py` — Paths for models, logs, and dumps; configurable `MODEL_ROOT`

## Key Parameters

When Debug mode is off (`DEBUG = False` in `main.py`), you can pass CLI args:

| Argument                          | Description                                             | Default |
| --------------------------------- | ------------------------------------------------------- | ------- |
| `--task`                          | Task name (e.g. `sst2`, `arc_e`, `arc_c`, `bbh`)        | —       |
| `--model_family` / `--model_size` | Model family and size (e.g. `opt` / `125m`)             | —       |
| `--kv_iter`                       | Number of forward iterations for KV refinement          | 1       |
| `--step_size`                     | Step size for gradient-free updates                     | 0.01    |
| `--exemplar_method`               | Few-shot sampling: `random`, `stratified`, or `written` | random  |
| `--quant_method`                  | Quantization: `deepspeed`, `8bit`, or `4bit`            | —       |

## Dependencies

Main dependencies: `torch`, `accelerate`, `transformers`, `deepspeed`, `datasets`, etc. Ensure these packages and a CUDA environment are installed.

## Models & Data

- Model paths are configured in `anchor.py` (`MODEL_ROOT`); models are loaded from the Hugging Face mirror by default.
- Datasets must be pre-downloaded; use `predownload_datasets.py` to fetch them in advance.

## Example Output

```plaintext
# python main.py

Run Prepared: kv15_default_eps0.01_beta0.9
        Task: sst2
        Logger save at /home/workspace/deep-thinking/logger/DEBUG/sst2/stratified1/seed42/opt/125m
Loading model with deepspeed inference engine...
Model loaded: /home/models/opt-125m
----------------------------------------------------------------------------------------------------
1 task(s) to evluate:
        [00] sst2
----------------------------------------------------------------------------------------------------
[ 1 /  1] Evaluate: sst2
Data loaded: inference.
Data loaded: sampling.
Multiple choice dataset: finish!
Num of choices: {2}
Tokenization finished: 872, max_length=68
* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ *
| Line | Text                                                                                               |
| 1    | Review: at every opportunity to do something clever                                                |
| 2    | Sentiment: positive                                                                                |
| 3    |                                                                                                    |
| 4    | Review: been discovered , indulged in and rejected as boring before i see this piece of crap again |
| 5    | Sentiment: negative                                                                                |
| 6    |                                                                                                    |
| 7    | <query starts from here>                                                                           |
* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ *
Running with batch size = 32
Iter=1   | {"lm_log_p": "52.5229", "norm_lm_log_p": "52.5229"}
Iter=2   | {"lm_log_p": "52.6376", "norm_lm_log_p": "52.6376"}
Iter=3   | {"lm_log_p": "52.6376", "norm_lm_log_p": "52.6376"}
Iter=4   | {"lm_log_p": "53.4404", "norm_lm_log_p": "53.4404"}
Iter=5   | {"lm_log_p": "60.8945", "norm_lm_log_p": "60.8945"}
Iter=6   | {"lm_log_p": "70.7569", "norm_lm_log_p": "70.7569"}
Iter=7   | {"lm_log_p": "73.8532", "norm_lm_log_p": "73.8532"}
Iter=8   | {"lm_log_p": "70.6422", "norm_lm_log_p": "70.6422"}
Iter=9   | {"lm_log_p": "65.1376", "norm_lm_log_p": "65.1376"}
Iter=10  | {"lm_log_p": "60.2064", "norm_lm_log_p": "60.2064"}
Iter=11  | {"lm_log_p": "57.7982", "norm_lm_log_p": "57.7982"}
Iter=12  | {"lm_log_p": "56.7661", "norm_lm_log_p": "56.7661"}
Iter=13  | {"lm_log_p": "57.3394", "norm_lm_log_p": "57.3394"}
Iter=14  | {"lm_log_p": "55.7339", "norm_lm_log_p": "55.7339"}
Iter=15  | {"lm_log_p": "54.5872", "norm_lm_log_p": "54.5872"}
           Performance Trending Monitor            
┏━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ idx ┃     acc ┃ acc_norm ┃ T[acc] ┃ T[acc_norm] ┃
┡━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│   1 │ 52.5229 │  52.5229 │ *      │ *           │
│   2 │ 52.6376 │  52.6376 │ **     │ **          │
│   3 │ 52.6376 │  52.6376 │        │             │
│   4 │ 53.4404 │  53.4404 │ ***    │ ***         │
│   5 │ 60.8945 │  60.8945 │ ****   │ ****        │
│   6 │ 70.7569 │  70.7569 │ *****  │ *****       │
│   7 │ 73.8532 │  73.8532 │ ****** │ ******      │
│   8 │ 70.6422 │  70.6422 │        │             │
│   9 │ 65.1376 │  65.1376 │        │             │
│  10 │ 60.2064 │  60.2064 │        │             │
│  11 │ 57.7982 │  57.7982 │        │             │
│  12 │ 56.7661 │  56.7661 │        │             │
│  13 │ 57.3394 │  57.3394 │        │             │
│  14 │ 55.7339 │  55.7339 │        │             │
│  15 │ 54.5872 │  54.5872 │        │             │
└─────┴─────────┴──────────┴────────┴─────────────┘

Task [sst2] => /home/workspace/deep-thinking/dump/DEBUG/sst2/stratified1/seed42/opt/125m/kv15_default_eps0.01_beta0.9/sst2.json
TL;DR => /home/workspace/deep-thinking/dump/DEBUG/sst2/stratified1/seed42/opt/125m/kv15_default_eps0.01_beta0.9/_tldr.json
```

## Citation

```bibtex
@inproceedings{yang2024iterative,
  title={Iterative forward tuning boosts in-context learning in language models},
  author={Yang, Jiaxi and Hui, Binyuan and Yang, Min and Wang, Bailin and Li, Bowen and Li, Binhua and Huang, Fei and Li, Yongbin},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={15460--15473},
  year={2024}
}
```
