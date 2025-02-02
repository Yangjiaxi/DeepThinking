from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="sst2",
    task_type=TaskType.Selection,
    suite=["sst", "classification", "retro"],
)
class SST2ProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["negative", "positive"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("glue", "sst2", "validation"),
            "sampling": ("glue", "sst2", "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["sentence"].strip(), "choices": self.CHOICES, "answer_idx": e["label"]})
        return data

    def promptify_input(self, query):
        with_query = f"Review: {query}\nSentiment:"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice


@register_task(
    name="sst5",
    task_type=TaskType.Selection,
    suite=["sst", "classification", "retro"],
)
class SST5ProbTask(SST2ProbTask):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = ["terrible", "negative", "neutral", "positive", "great"]
        self.can_be_stratified = True
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("SetFit/sst5", None, "validation"),
            "sampling": ("SetFit/sst5", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["text"].strip(), "choices": self.CHOICES, "answer_idx": e["label"]})
        return data
