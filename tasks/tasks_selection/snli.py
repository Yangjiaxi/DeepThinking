from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="snli",
    task_type=TaskType.Selection,
    suite=["classification", "retro2"],
)
class SNLICProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = True
        self.CHOICES = ["Yes", "Maybe", "No"]
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("snli", None, "validation"),  # validation
            "sampling": ("snli", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []

        for e in raw_data:
            # label == 0 -> entailment      -> Yes
            # label == 1 -> neutral         -> Maybe
            # label == 2 -> contradiction   -> No
            if e["label"] == -1:
                continue

            assert e["label"] in list(range(len(self.CHOICES)))
            data.append(
                {
                    "query": (e["premise"], e["hypothesis"]),
                    "choices": self.CHOICES,
                    "answer_idx": e["label"],  # there's -1 label
                }
            )
        return data

    # def promptify_input(self, query):
    #     text1, text2 = query
    #     with_query = f'{text1} Can we say "{text2}"?'
    #     return with_query

    # def promptify_input_with_choice(self, query, choice):
    #     with_query_and_choice = f"{self.promptify_input(query)} {choice}"
    #     return with_query_and_choice

    def promptify_input(self, query):
        text1, _ = query
        with_query = f"{text1}?"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        _, text2 = query
        answer = f"{choice}, {text2}"
        with_query_and_choice = f"{self.promptify_input(query)} {answer}"
        return with_query_and_choice
