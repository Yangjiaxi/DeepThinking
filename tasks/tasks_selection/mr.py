from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="mr",
    task_type=TaskType.Selection,
    suite=["classification", "retro"],
)
class MRProbTask(BaseProbTaskManager):  # from rotten tomatoes, movie review dataest
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["negative", "positive"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("rotten_tomatoes", None, "validation"),
            "sampling": ("rotten_tomatoes", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["text"].strip(), "choices": self.CHOICES, "answer_idx": e["label"]})
        return data

    def promptify_input(self, query):
        with_query = f"Review: {query}\nSentiment:"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice
