from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="subj",
    task_type=TaskType.Selection,
    suite=["classification", "retro"],
)
class SubJProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["negative", "positive"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("SetFit/subj", None, "test"),
            "sampling": ("SetFit/subj", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append(
                {
                    "query": e["text"].strip(),
                    "choices": self.CHOICES,
                    "answer_idx": e["label"],
                }
            )
        return data

    def promptify_input(self, query):
        with_query = f"Input: {query}\nType:"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice
