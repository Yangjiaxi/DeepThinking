from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="qnli",
    task_type=TaskType.Selection,
    suite=["classification", "retro2"],
)
class QNLICProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = True
        self.CHOICES = ["Yes", "No"]
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("SetFit/qnli", None, "validation"),  # validation
            "sampling": ("SetFit/qnli", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            # label == 0 -> entailment      -> Yes
            # label == 1 -> not entailment  -> No
            data.append(
                {
                    "query": (e["text1"], e["text2"]),
                    "choices": self.CHOICES,
                    "answer_idx": e["label"],
                }
            )
        return data

    def promptify_input(self, query):
        text1, text2 = query
        with_query = f'{text1} Can we know "{text2}"?'
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice
