import gc
import json
import logging
import os
import textwrap
import warnings

import torch
from accelerate import find_executable_batch_size
from accelerate.utils import gather_object
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm_rich as tqdm

from common import AdvantageLogger

from models.meta_optimizer import AttnOptimWrapper
from models.meta_optimizer_norm import AttnOptimWrapper as AttnOptimWrapperNorm

from utils.logger import tabular_pretty_print

# from tqdm import tqdm
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

logger = logging.getLogger("task")


def example_showcase(exemplar_str):
    try:
        text_width = min(os.get_terminal_size().columns - 30, 150)
    except:
        text_width = 150

    exemplar_showcase = [["Line", "Text"]]
    for line_idx, line in enumerate(exemplar_str.split("\n")):
        if len(line) > text_width:
            splitted_lines = textwrap.wrap(line, text_width)
            exemplar_showcase.append([str(line_idx + 1), splitted_lines[0]])
            for remained in splitted_lines[1:]:
                exemplar_showcase.append(["", remained])
        else:
            exemplar_showcase.append([str(line_idx + 1), line])

    exemplar_showcase[-1][-1] += "<query starts from here>"
    for line in tabular_pretty_print(exemplar_showcase):
        logger.info(line)


def the_shape(pack):
    if isinstance(pack, (list, tuple)):
        return f"{len(pack)} * {the_shape(pack[0])}"
    if isinstance(pack, torch.Tensor):
        return pack.size()


@torch.no_grad()
def do_infer_probs(model, exemplar_attn_kv, exemplar_attn_mask, batched_choices_input):
    batched_choices_logprobs = []
    for batched_one_choice_input in batched_choices_input:
        batch_input_ids, batch_attention_mask, batch_choice_start, batch_choice_end = batched_one_choice_input
        # print(f"{the_shape(batch_input_ids) = }")
        bs = len(batch_input_ids)

        merged_attn_mask = torch.cat((exemplar_attn_mask.expand(bs, -1), batch_attention_mask), dim=1)
        # [B, #Heads, Length, Hidden]
        # print(f"L64 | {the_shape(exemplar_attn_kv)}")
        expand_exemplar_attn_kv = [
            [layer_k.expand((bs, -1, -1, -1)), layer_v.expand((bs, -1, -1, -1))]
            for layer_k, layer_v in exemplar_attn_kv
        ]
        # print(f"L66 | {the_shape(expand_exemplar_attn_kv)}")

        batched_logits = model(
            input_ids=batch_input_ids,  # [B, L']
            attention_mask=merged_attn_mask,  # [B, L + L']
            past_key_values=expand_exemplar_attn_kv,  # num_layers * 2 * [B, num_heads, L, H]
        ).logits
        batched_output = F.log_softmax(batched_logits, dim=-1)  # [B, L', Vocab]

        batched_one_choice_logprobs = []
        for input_ids, choice_start, choice_end, lm_logprobs in zip(
            batch_input_ids, batch_choice_start, batch_choice_end, batched_output
        ):
            choice_tokens = input_ids[choice_start:choice_end].unsqueeze(1)  # [L, 1]
            choice_logprobs = lm_logprobs[choice_start - 1 : choice_end - 1]  # [L, Vocab]

            extracted = torch.gather(choice_logprobs, -1, choice_tokens).squeeze(-1)

            choice_length = choice_end - choice_start
            lm_log_p = torch.sum(extracted).item()
            norm_lm_log_p = (lm_log_p / choice_length).item()

            choice_lm_info = {"lm_log_p": lm_log_p, "norm_lm_log_p": norm_lm_log_p}
            batched_one_choice_logprobs.append(choice_lm_info)
        batched_choices_logprobs.append(batched_one_choice_logprobs)
    return batched_choices_logprobs


def run_selective_task(model, tokenizer, accelerator, task_agent_cls, args):
    task_agent = task_agent_cls(args.prompt_version)
    task_agent.set_seed(args.seed)
    # with accelerator.main_process_first():
    task_agent.do_load()
    # accelerator.wait_for_everyone()

    dataset = task_agent.make_inference_dataset(tokenizer)

    if args.exemplar_method == "random":
        exemplar_str = task_agent.random_selected_exemplars(args.num_fewshot)
    elif args.exemplar_method == "stratified":
        exemplar_str = task_agent.stratified_sampling(args.num_stratified)
    elif args.exemplar_method == "written":
        exemplar_str = task_agent.handcrafted_exemplars()
    else:
        raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

    example_showcase(exemplar_str)
    exemplar_input_ids, exemplar_attn_mask = [e.cuda() for e in dataset.tokenize_demonstration(exemplar_str)]
    # print(f"{the_shape(exemplar_input_ids) = }")
    # print(f"{the_shape(exemplar_attn_mask) = }")

    if args.model_family == "gpt2":
        THRESHOLD = 400
        if len(exemplar_input_ids) > THRESHOLD:
            ori_len = len(exemplar_input_ids)
            exemplar_input_ids = exemplar_input_ids[-THRESHOLD:]
            exemplar_attn_mask = exemplar_attn_mask[-THRESHOLD:]
            logger.info(f"Truncate: {ori_len} -> {len(exemplar_input_ids)}")

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def run_with_batch_size_retry(batch_size):
        logger.info(f"Running with batch size = {batch_size}")
        loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=8)
        loader = accelerator.prepare(loader)

        meta_optim = AttnOptimWrapper(model, step_size=args.step_size, momentum=args.momentum)
        meta_optim.init()
        trace_logger = AdvantageLogger()
        trace_logger.set_value_names("acc", "acc_norm")
        for idx in range(args.kv_iter):
            exemplar_kv = meta_optim.step(exemplar_input_ids)
            # print(f"{idx=}, {the_shape(exemplar_kv) = }")

            generated_info = []  # question * [choice0_prob, choice1_prob]
            for batch_input in tqdm(
                loader,
                total=len(loader),
                disable=not accelerator.is_main_process,
                leave=False,
            ):
                batch_input = [[e.cuda() for e in batch_choice] for batch_choice in batch_input]
                batch_output = do_infer_probs(
                    model,
                    exemplar_kv,
                    exemplar_attn_mask.unsqueeze(0),
                    batch_input,
                )  # [batch_of_choice0, batch_of_choice1, ...]

                zipped_logprobs = list(zip(*batch_output))  # batch * (choice0, choice1, ...)
                zipped_logprobs = gather_object(zipped_logprobs)
                if accelerator.is_main_process:
                    generated_info.extend(zipped_logprobs)

            if accelerator.is_main_process:
                full_info, metric = task_agent.post_process(generated_info, metric_output=False)
                metric_s = json.dumps(metric, indent=None)
                logger.info(f"Iter={idx+1: <3} | {metric_s}")
                trace_logger.submit(idx + 1, metric["lm_log_p"], metric["norm_lm_log_p"])
                gc.collect()
        return trace_logger

    try:
        trace_logger = run_with_batch_size_retry()
        if accelerator.is_main_process:
            for line in trace_logger.pretty_print():
                logger.info(line)
            return trace_logger.summary()
        else:
            return {}
    except RuntimeError as e:
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        return None  # maybe no executable batchsize found


def run_selective_task_norm(model, tokenizer, accelerator, task_agent_cls, args):
    task_agent = task_agent_cls(args.prompt_version)
    task_agent.set_seed(args.seed)
    # with accelerator.main_process_first():
    task_agent.do_load()
    # accelerator.wait_for_everyone()

    dataset = task_agent.make_inference_dataset(tokenizer)

    if args.exemplar_method == "random":
        exemplar_str = task_agent.random_selected_exemplars(args.num_fewshot)
    elif args.exemplar_method == "stratified":
        exemplar_str = task_agent.stratified_sampling(args.num_stratified)
    elif args.exemplar_method == "written":
        exemplar_str = task_agent.handcrafted_exemplars()
    else:
        raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

    example_showcase(exemplar_str)
    exemplar_input_ids, exemplar_attn_mask = [e.cuda() for e in dataset.tokenize_demonstration(exemplar_str)]

    batch_size = args.batch_size
    logger.info(f"Running with batch size = {batch_size}")
    loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=8)
    loader = accelerator.prepare(loader)

    meta_optim = AttnOptimWrapperNorm(model, step_size=args.step_size, momentum=args.momentum)
    meta_optim.init()
    trace_logger = AdvantageLogger()
    trace_logger.set_value_names("acc", "acc_norm")
    metric_store = []

    for idx in range(args.kv_iter):
        exemplar_kv = meta_optim.step(exemplar_input_ids)
        kv_diff_norm = meta_optim.kv_diff_norm
        diff_ks, diff_vs = zip(*kv_diff_norm)

        diff_k_data = [f"{diff_k.item():.3f}" for diff_k in diff_ks]
        diff_v_data = [f"{diff_v.item():.3f}" for diff_v in diff_vs]

        generated_info = []  # question * [choice0_prob, choice1_prob]
        for batch_input in tqdm(
            loader,
            total=len(loader),
            disable=not accelerator.is_main_process,
            leave=False,
        ):
            batch_input = [[e.cuda() for e in batch_choice] for batch_choice in batch_input]
            batch_output = do_infer_probs(
                model,
                exemplar_kv,
                exemplar_attn_mask.unsqueeze(0),
                batch_input,
            )  # [batch_of_choice0, batch_of_choice1, ...]

            zipped_logprobs = list(zip(*batch_output))  # batch * (choice0, choice1, ...)
            zipped_logprobs = gather_object(zipped_logprobs)
            if accelerator.is_main_process:
                generated_info.extend(zipped_logprobs)

        if accelerator.is_main_process:
            full_info, metric = task_agent.post_process(generated_info, metric_output=False)
            metric_s = json.dumps(metric, indent=None)
            logger.info(f"Iter={idx+1: <3} | {metric_s}")
            trace_logger.submit(idx + 1, metric["lm_log_p"], metric["norm_lm_log_p"])
            gc.collect()
            metric_store.append({"iter": idx + 1, "data": metric, "diff_k": diff_k_data, "diff_v": diff_v_data})

    if accelerator.is_main_process:
        return metric_store
    else:
        return {}


def run_generative_task(dataset, model, kwargs):
    pass
