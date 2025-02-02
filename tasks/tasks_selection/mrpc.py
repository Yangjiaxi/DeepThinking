from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="mrpc",
    task_type=TaskType.Selection,
    suite=["classification", "retro2"],
)
class MRPCCProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = True
        self.CHOICES = ["No", "Yes"]
        self.num_base_shot = len(self.CHOICES)

    def dataset_signature(self):
        return {
            "inference": ("SetFit/mrpc", None, "validation"),  # test
            "sampling": ("SetFit/mrpc", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            # label == 1 -> equivalent     -> Yes
            # label == 0 -> not equivalent -> No
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
        with_query = f'{text1} Can we say "{text2}"?'
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice

    # def promptify_input(self, query):
    #     text1, _ = query
    #     with_query = f" {text1}"
    #     return with_query

    # def promptify_input_with_choice(self, query, choice):
    #     _, text2 = query
    #     if choice == "No":
    #         answer = f"We can't say {text2}"
    #     elif choice == "Yes":
    #         answer = f"We can say {text2}"
    #     with_query_and_choice = f"{self.promptify_input(query)} {answer}"
    #     return with_query_and_choice
