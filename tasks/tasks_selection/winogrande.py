from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="winogrande",
    task_type=TaskType.Selection,
    suite=["multiple-choice"],
)
class WinoGrandeProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def dataset_signature(self):
        return {
            # "inference": ("winogrande", "winogrande_xs", "validation"),
            # "sampling": ("winogrande", "winogrande_xs", "train"),
            "inference": ("winogrande", "winogrande_xl", "validation"),
            "sampling": ("winogrande", "winogrande_xs", "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = e["sentence"]
            choices = [e["option1"], e["option2"]]
            label = int(e["answer"]) - 1
            data.append({"query": query, "choices": choices, "answer_idx": label})
        return data

    def promptify_input(self, query):
        before_under, after_under = [e.strip() for e in query.split("_")]
        with_query = before_under
        return with_query

    def promptify_input_with_choice(self, query, choice):
        before_under, after_under = [e.strip() for e in query.split("_")]
        with_query = before_under
        with_query_and_choice = f"{with_query} {choice} {after_under}"
        return with_query_and_choice
