import random

from tasks import register_task, TaskType
from tasks.base_selection import BaseProbTaskManager


@register_task(
    name="agnews",
    task_type=TaskType.Selection,
    suite=["classification", "retro"],
)
class AGNewsProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["World", "Sports", "Business", "Technology"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("ag_news", None, "test"),
            "sampling": ("ag_news", None, "train"),
        }

    def do_load(self):
        with self.deterministic_context():
            full_result_data = self.do_load_part("inference")
            self.raw_data_inference = random.sample(full_result_data, 2000)
            self.raw_data_sampling = self.do_load_part("sampling")

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["text"].strip(), "choices": self.CHOICES, "answer_idx": e["label"]})
        return data

    def promptify_input(self, query):
        with_query = f"Article: {query}\nCategory:"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice
