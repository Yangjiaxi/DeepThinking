import random

from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="csqa1",
    task_type=TaskType.Selection,
    suite=["multiple-choice", "retro2"],
)
class CSQA1ProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 5

    def dataset_signature(self):
        return {
            "inference": ("commonsense_qa", None, "validation"),
            "sampling": ("commonsense_qa", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append(
                {
                    "query": e["question"],
                    "choices": e["choices"]["text"],
                    "answer_idx": ord(e["answerKey"]) - ord("A"),
                }
            )
        return data

    def promptify_input(self, query):
        with_query = f"{query}"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice
