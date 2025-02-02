import logging
import random
from collections import defaultdict

import numpy as np

from tasks.base import BaseTaskManager
from tasks.loader import TokenizedForMCRightPad

logger = logging.getLogger("task")


class BaseProbTaskManager(BaseTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.CHOICES = None

        self.saved_examples = None

    def stratified_sampling(self, num_k_shots):
        num_shots = self.num_base_shot * num_k_shots

        if not self.can_be_stratified:
            logger.info("Cannot be stratified, fallback to random selection.")
            return self.random_selected_exemplars(num_shots)

        ans_set = set(e["answer_idx"] for e in self.raw_data_sampling)
        ans_map = defaultdict(list)
        for idx, e in enumerate(self.raw_data_sampling):
            label = e["answer_idx"]
            ans_map[label].append(idx)

        per_label = num_shots // len(ans_set)
        residual = num_shots - per_label * len(ans_set)

        selected_ids = []
        with self.deterministic_context():
            for label, all_ids in ans_map.items():
                selected = random.sample(all_ids, per_label)
                selected_ids.extend(selected)

            remain_ids = set(range(len(self.raw_data_sampling))) - set(selected_ids)
            residual_selected = random.sample(remain_ids, residual)
            selected_ids.extend(residual_selected)
            random.shuffle(selected_ids)

        selected_examples = [self.raw_data_sampling[i] for i in selected_ids]

        self.saved_examples = selected_examples
        return self.build_exemplar_from_examples(self.task_prefix, selected_examples)

    def promptify_input(self, query):
        # not likely being used under multiple-choice tasks
        raise NotImplementedError

    def promptify_input_with_choice(self, query, choice):
        raise NotImplementedError

    def promptify_golden(self, example):
        if isinstance(example, dict):  # from random / stratified
            golden_idx = example["answer_idx"]
            query, golden_choice = example["query"], example["choices"][golden_idx]
        elif isinstance(example, tuple):  # from handcrafted
            query, golden_choice = example
        return self.promptify_input_with_choice(query, golden_choice)

    def make_inference_dataset(self, tokenizer):
        def mc_prompt_fn(query, choice):
            return (
                self.promptify_input(query),
                self.promptify_input_with_choice(query, choice),
            )

        return TokenizedForMCRightPad(
            self.raw_data_inference,
            tokenizer,
            mc_prompt_fn,  # (query, choice) -> P(query), P(query, choice)
        )

    @staticmethod
    def merge_choice_info(choice_info):
        merged = {}
        for k in ["lm_log_p", "norm_lm_log_p"]:
            one_metric_merged = []
            for info in choice_info:
                one_metric_merged.append(info[k])
            merged[k] = one_metric_merged
        return merged

    @staticmethod
    def choice_info_to_predictions(info):
        lm_log_p_idx = int(np.argmax(info["lm_log_p"]))
        norm_lm_log_p_idx = int(np.argmax(info["norm_lm_log_p"]))
        return {"lm_log_p": lm_log_p_idx, "norm_lm_log_p": norm_lm_log_p_idx}

    def post_process(self, generated_info, metric_output=True):
        full_info = []
        num_tested = 0
        num_correct = {"lm_log_p": 0, "norm_lm_log_p": 0}
        for idx, (data, choice_info) in enumerate(zip(self.raw_data_inference, generated_info)):
            merged_choice_info = self.merge_choice_info(choice_info)
            merged_predictions_idx = self.choice_info_to_predictions(merged_choice_info)
            combined = {
                "_id": idx,
                "choice_logprob": merged_choice_info,
                "predicted": merged_predictions_idx,
                **data,  # query & answer_idx
            }
            num_tested += 1
            ground_idx = combined["answer_idx"]
            for k in num_correct:
                num_correct[k] += 1 if merged_predictions_idx[k] == ground_idx else 0
            full_info.append(combined)

        if metric_output:
            logger.info("v" * 30)
            for k in num_correct:
                t = num_correct[k] * 100 / num_tested
                logger.info(f"Acc @ {k} : {num_correct[k]} / {num_tested} = {t:.4f}")
            logger.info("^" * 30)

        acc_info = {k: f"{(v * 100 / num_tested):.4f}" for k, v in num_correct.items()}
        return full_info, acc_info
