import random

from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="hellaswag",
    task_type=TaskType.Selection,
    suite=["multiple-choice"],
)
class HellaSwagProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def dataset_signature(self):
        return {
            "inference": ("hellaswag", None, "validation"),
            "sampling": ("hellaswag", None, "train"),
        }

    def do_load(self):
        with self.deterministic_context():
            full_result_data = self.do_load_part("inference")
            self.raw_data_inference = random.sample(full_result_data, 2000)
            self.raw_data_sampling = self.do_load_part("sampling")

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["ctx"], "choices": e["endings"], "answer_idx": int(e["label"])})
        return data

    def promptify_input(self, query):
        with_query = f"{query}"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice
