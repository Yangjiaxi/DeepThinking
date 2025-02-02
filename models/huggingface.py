import logging

import deepspeed
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("task")


def load_model(
    accelerator,
    model_name_or_path,
    quant_method,
):
    if quant_method == "deepspeed":
        logger.info("Loading model with deepspeed inference engine...")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        ds_engine = deepspeed.init_inference(model, dtype=torch.bfloat16)
        model = ds_engine.module
    elif quant_method == "8bit":
        logger.info("Loading model in 8bit...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            device_map={"": accelerator.process_index},
        )
    elif quant_method == "4bit":
        logger.info("Loading model in 4bit...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            device_map={"": accelerator.process_index},
        )
    else:
        logger.info("Loading model in bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    accelerator.wait_for_everyone()
    # model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
    )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
        else:
            raise ValueError("No eos_token or bos_token found")

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def auto_set_model(model):
    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=["GPTNeoXLayer", "GPTNeoXMLP"],
        dtype="bfloat16",
        low_zero=False,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["GPTNeoXLayer", "GPTNeoXMLP"],
        dtype="bfloat16",
    )
    model = dispatch_model(model, device_map=device_map)
    return model
