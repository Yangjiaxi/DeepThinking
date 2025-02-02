import argparse
import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from anchor import dump_root, logger_root
from tasks import list_all_tasks
from utils.logger import fmt_float, tabular_pretty_print


def setup_plain_seed(SEED):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def setup_seed(SEED):
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    setup_plain_seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_gpu(gpu_s):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_s)


def setup_env(gpu_s, seed):
    setup_gpu(gpu_s)
    setup_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def mk_parser():
    psr = argparse.ArgumentParser(add_help=False)
    psr.add_argument("--dump_root", type=Path, default=dump_root)
    psr.add_argument("--logger_root", type=Path, default=logger_root)

    psr.add_argument("--seed", type=int, default=42)
    psr.add_argument("--prompt_version", type=str, default="v1")
    psr.add_argument("--task", type=str, choices=list_all_tasks())
    psr.add_argument("--data_file", type=str)

    # [Load from pre-downloaded folder] --model /path/to/model/checkpoint or signature
    psr.add_argument("--model", type=str, default=None)
    psr.add_argument("--model_family", type=str, default=None)
    psr.add_argument("--model_size", type=str, default=None)

    psr.add_argument("--batch_size", type=int, default=0)  # 0 for auto-detect, -1 for FORCE auto-detect
    psr.add_argument("--quant_method", type=str, default=None, choices=["deepspeed", "8bit", "4bit"])
    psr.add_argument("--no_console", action="store_true", default=False)

    psr.add_argument("--exemplar_method", type=str, default="random", choices=["random", "stratified", "written"])
    psr.add_argument("--num_stratified", type=int, default=1)
    psr.add_argument("--num_fewshot", type=int, default=1)

    psr.add_argument("--kv_iter", type=int, default=1)
    psr.add_argument("--step_size", type=float, default=0.01)
    psr.add_argument("--momentum", type=float, default=0.9)
    return psr


class GridMetric:
    def __init__(self, grid_size, decimal=1):
        self.data = np.zeros((grid_size, grid_size), dtype=float)
        self.format_f = np.vectorize(lambda x: fmt_float(x, decimal))

    def submit(self, i, j, metric):
        # i, j starts from 0
        # 0 <= i,j < grid_size
        self.data[i][j] = metric

    def pretty_print(self):
        for line in tabular_pretty_print(self.format_f(self.data).tolist()):
            yield line


class AdvantageLogger:
    def __init__(self):
        self.log = []
        self.cur_best = 0.0
        self.is_better = np.greater
        self.value_names = ["acc", "norm_acc"]
        self.cached_summary = None

    def set_value_names(self, *names):
        self.value_names = names

    def submit(self, idx, *values):
        values = [float(e) for e in values]
        self.log.append((idx, *values))

    def pretty_print(self):
        table = Table(title="Performance Trending Monitor")
        table.add_column("idx", justify="right")
        for col_name in self.value_names:
            table.add_column(col_name, justify="right", no_wrap=True)
        for col_name in self.value_names:
            trend_name = escape(f"T[{col_name}]")
            table.add_column(trend_name, justify="left", no_wrap=True)

        sign_meter = {name: {"best": {"score": 0.0, "at": 0}, "start": None, "_sign": ""} for name in self.value_names}

        for idx, *values in self.log:
            markers = []
            for name, v in zip(self.value_names, values):
                if sign_meter[name]["start"] is None:
                    sign_meter[name]["start"] = {"score": v, "at": idx}

                if self.is_better(v, sign_meter[name]["best"]["score"]):
                    sign_meter[name]["best"]["at"] = idx
                    sign_meter[name]["best"]["score"] = v
                    sign_meter[name]["_sign"] += "*"
                    markers.append(sign_meter[name]["_sign"])
                else:
                    markers.append("")

            table.add_row(str(idx), *[str(e) for e in values], *markers)

        self.cached_summary = sign_meter

        try:
            text_width = os.get_terminal_size().columns - 30
        except:
            text_width = 150
        console = Console(file=io.StringIO(), width=text_width)
        console.print(table)
        table_repr = console.file.getvalue()
        for line in table_repr.split("\n"):
            yield line

    def summary(self):
        tldr_d = {}
        for name in self.value_names:
            d = self.cached_summary[name]
            tldr = f"[{d['start']['at']}] {d['start']['score']} --> [{d['best']['at']}] {d['best']['score']}"
            tldr_d[name] = tldr

        data_d = []
        for log in self.log:
            idx, *values = log
            data_d.append({"idx": idx, **{name: v for name, v in zip(self.value_names, values)}})
        return {"tldr": tldr_d, "data": data_d}
