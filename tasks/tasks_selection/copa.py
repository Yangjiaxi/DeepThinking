from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="copa",
    task_type=TaskType.Selection,
    suite=["multiple-choice"],
)
class COPAProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def dataset_signature(self):
        return {
            "inference": ("super_glue", "copa", "validation"),
            "sampling": ("super_glue", "copa", "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = (e["premise"], e["question"])
            choices = [e["choice1"], e["choice2"]]
            data.append({"query": query, "choices": choices, "answer_idx": e["label"]})
        return data

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("v1"):
            premise, question = query
            if premise.endswith("."):
                premise = premise[:-1]  # looks like a sentence

            intermediate = "because" if question == "cause" else "therefore"
            with_query = f"{premise} {intermediate}"
            with_query_and_choice = f"{with_query} {choice}"
        else:
            raise ValueError(f"COPA: Not supported prompt_version: {self.prompt_version}")
        return with_query, with_query_and_choice
