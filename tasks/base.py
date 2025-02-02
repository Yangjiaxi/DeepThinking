import json
import logging
import random
import re

import datasets

from anchor import hf_datasets_root
from utils.rng_ctx import EmptyContext, RandomContext

logger = logging.getLogger("task")

HAND_WRITTEN_SHUFFLE = True


class BaseTaskManager:
    def __init__(self, prompt_version):
        if prompt_version == "default":
            self.prompt_version = self.default_prompt_version()
        else:
            self.prompt_version = prompt_version

        self.raw_data_inference = None
        self.raw_data_sampling = None

        self._seed = None

    def default_prompt_version(self):
        return "v1"

    def deterministic_context(self):
        if self._seed is None:
            logger.info("Please use .set_seed to avoid non-deterministic behavior introduced by random operations.")
            return EmptyContext()

        return RandomContext(seed=self._seed)

    def set_seed(self, seed):
        self._seed = seed

    def dataset_signature(self):
        # {
        #      "inference":  (dataset_name, subset, split),  # which produce the final result
        #      "sampling": (dataset_name, subset, split),  # which we sample ICL few-shot examples
        # }
        raise NotImplementedError

    @property
    def task_prefix(self):
        # maybe instruction, default to empty string
        return ""

    def dataset_part(self, part):
        return self.dataset_signature()[part]

    def dataset_preprocess(self, raw_data):
        raise NotImplementedError

    def exemplar_seperator(self):
        return "\n\n"

    def task_prepared_examplars(self):
        return NotImplementedError

    def handcrafted_exemplars(self):
        examplars = self.task_prepared_examplars()
        if HAND_WRITTEN_SHUFFLE:
            with self.deterministic_context():
                selected_examples = random.sample(examplars, len(examplars))
        else:
            selected_examples = examplars

        return self.build_exemplar_from_examples(self.task_prefix, selected_examples)

    def random_selected_exemplars(self, num_shots):
        with self.deterministic_context():
            selected_examples = random.sample(self.raw_data_sampling, num_shots)

        return self.build_exemplar_from_examples(self.task_prefix, selected_examples)

    def promptify_golden(self, example):
        raise NotImplementedError

    def build_exemplar_from_examples(self, prefix, selected_examples):
        s = prefix
        if len(s):
            s += self.exemplar_seperator()

        for example in selected_examples:
            line = self.promptify_golden(example)
            s += line + self.exemplar_seperator()
        return s

    def dataset_file_path(self, part):
        part_pack = self.dataset_part(part)
        if part_pack is None:
            return None  # return None when `part` contains a None tuple

        dataset_name, subset, split = part_pack
        dumped_folder = hf_datasets_root.joinpath("dumped")
        if not dumped_folder.exists():
            dumped_folder.mkdir(parents=True)

        file_name = f"{dataset_name}-{subset}-{split}.jsonl"
        file_name = re.sub(r"[^\w_. -]", "_", file_name)
        return dumped_folder.joinpath(file_name)

    def do_load_part(self, part):
        f_path = self.dataset_file_path(part)
        if f_path is None:
            return None  # return None when `part` contains a None tuple

        if not f_path.exists():
            self.not_exist_download(part)
            return self.do_load_part(part)  # call once more
        else:
            with f_path.open("r") as f:
                raw_data = [json.loads(line) for line in f]
            data = self.dataset_preprocess(raw_data)
            logger.info(f"Data loaded: {part}.")
            return data

    def do_load(self):
        self.raw_data_inference = self.do_load_part("inference")
        self.raw_data_sampling = self.do_load_part("sampling")

    def not_exist_download(self, part):
        f_path = self.dataset_file_path(part)
        logger.info(f"{f_path} not exist, download from huggingface datasets hub...")

        dataset_name, subset, split = self.dataset_part(part)
        data = self.do_download(dataset_name, subset, split=split, cache_dir=str(hf_datasets_root))
        data.to_json(f_path)
        logger.info(f"... success, saved at: {f_path}")

    @staticmethod
    def do_download(dataset_name, subset, split, cache_dir):
        raw_data = datasets.load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
        logger.info("Download success.")
        return raw_data

    def make_inference_dataset(self, tokenizer):
        raise NotImplementedError

    def post_process(self, generated_info, metric_output=True):
        raise NotImplementedError
