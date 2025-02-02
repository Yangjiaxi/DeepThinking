import gc
import json
import logging
from pathlib import Path

from accelerate import Accelerator
import torch

from anchor import model_to_ckpt
from common import mk_parser, setup_seed
from core import run_selective_task
from models import load_model
from tasks import TaskType, get_registered_task
from utils.logger import setup_logger
from utils.tools import ensure_folder

logger = logging.getLogger("task")


if __name__ == "__main__":
    DEBUG = True

    parser = mk_parser()
    if DEBUG:
        fake_cmd = (
            "--prompt_version default "
            "--task sst2 "
            "--exemplar_method stratified --num_stratified 1 "
            "--model_family opt --model_size 125m "
            "--kv_iter 15 "
            "--step_size 0.01 "
            "--batch_size 32 "
            "--seed 42 "
            "--quant_method deepspeed"
        )
        args = parser.parse_args(fake_cmd.strip().split())
    else:
        args = parser.parse_args()

    accelerator = Accelerator()

    if DEBUG:
        args.logger_root = args.logger_root.joinpath("DEBUG")
        args.dump_root = args.dump_root.joinpath("DEBUG")

    if args.model is None:
        if not args.model_family or not args.model_size:
            raise ValueError(f"Please privide `args.model` or `args.model_family + args.model_size`")

        args.model = model_to_ckpt[args.model_family][args.model_size]
        model_abbr = Path(args.model_family).joinpath(args.model_size)
    else:
        maybe_path = Path(args.model)
        if Path(args.model).exists():
            model_abbr = maybe_path.relative_to(list(maybe_path.parents)[1])
        else:
            model_abbr = maybe_path

    run_name = f"kv{args.kv_iter}"
    run_name += f"_{args.prompt_version}"
    run_name += f"_eps{args.step_size}_beta{args.momentum}"

    exemplar_abbr = args.exemplar_method
    if args.exemplar_method == "stratified":
        exemplar_abbr += f"{args.num_stratified}"
    elif args.exemplar_method == "random":
        exemplar_abbr += f"{args.num_fewshot}"
    else:  # args.exemplar_method == 'written':
        pass

    rest_folder = Path(args.task).joinpath(exemplar_abbr).joinpath(f"seed{args.seed}").joinpath(model_abbr)

    logger_folder = args.logger_root.joinpath(rest_folder)
    dump_folder = args.dump_root.joinpath(rest_folder).joinpath(run_name)

    tldr_file = dump_folder.joinpath(f"_tldr.json")

    if accelerator.is_main_process:
        ensure_folder(logger_folder, parents=True)
        ensure_folder(dump_folder, parents=True)

    setup_seed(args.seed)
    setup_logger(
        logger_folder,
        log_file_name=f"{run_name}.log",
        console_output=not args.no_console,
        enabled=accelerator.is_main_process,
    )
    if not DEBUG and tldr_file.exists():
        logger.info("All tasks done, exit: {tldr_file}")
        exit(0)

    logger.info(f"Run Prepared: {run_name}")
    logger.info(f"\tTask: {args.task}")
    logger.info(f"\tLogger save at {logger_folder}")

    # 1. load model, tokenizer
    model, tokenizer = load_model(
        accelerator,
        args.model,
        quant_method=args.quant_method,
    )
    torch.autograd.set_grad_enabled(False)

    logger.info(f"Model loaded: {args.model}")

    tldr_collected = {
        "_failed": [],
    }
    loaded_tasks = get_registered_task(args.task)  # -> {`task_name`: (task_fn, task_type)}
    logger.info("-" * 100)
    logger.info(f"{len(loaded_tasks)} task(s) to evluate:")
    for idx, task_name in enumerate(loaded_tasks.keys()):
        logger.info(f"\t[{idx:02d}] {task_name}")
    logger.info("-" * 100)

    original_batch_size = args.batch_size
    for task_idx, (task_name, (TaskHandler, task_type)) in enumerate(loaded_tasks.items()):
        logger.info(f"[{task_idx + 1:2d} / {len(loaded_tasks):2d}] Evaluate: {task_name}")
        task_result_file = dump_folder.joinpath(f"{task_name}.json")
        if not DEBUG and task_result_file.exists():
            with task_result_file.open("r") as f:
                maybe_done = json.load(f)
            if "tldr" in maybe_done:
                tldr_collected[task_name] = maybe_done["tldr"]
                logger.info(f"\t[{task_name}] already done, skip: {task_result_file}")
                continue

        if task_type == TaskType.Selection:
            if task_name in ["high_school_european_history", "world_religions"]:  # weird tasks
                args.batch_size = 1
            else:
                args.batch_size = original_batch_size
            task_ret = run_selective_task(model, tokenizer, accelerator, TaskHandler, args)
        else:  # task_type == TaskType.Generation
            # TODO maybe in the future
            pass

        if task_ret is not None:
            if accelerator.is_main_process:
                with task_result_file.open("w") as f:
                    json.dump(task_ret, f, indent=2)
                logger.info(f"Task [{task_name}] => {task_result_file}")
                tldr_collected[task_name] = task_ret["tldr"]
        else:
            tldr_collected["_failed"].append(task_name)
            logger.info(f"\t[{task_name}] failed. Now len(failed)={len(tldr_collected['_failed'])}")

        gc.collect()

    if accelerator.is_main_process:
        with tldr_file.open("w") as f:
            json.dump(tldr_collected, f, indent=2)
        logger.info(f"TL;DR => {tldr_file}")

    del model
