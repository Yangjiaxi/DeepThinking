from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="obqa",
    task_type=TaskType.Selection,
    suite=["multiple-choice"],
)
class OBQAProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def dataset_signature(self):
        return {
            "inference": ("openbookqa", None, "validation"),
            "sampling": ("openbookqa", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"query": e["question_stem"], "choices": e["choices"]["text"], "answer_idx": ord(e["answerKey"]) - ord("A")})
        return data

    def multiple_choice_promptify(self, query, choice):
        if self.prompt_version.startswith("v1"):
            with_query = query
            with_query_and_choice = f"{with_query} {choice}."
        else:
            raise ValueError(f"OBQA: Not supported prompt_version: {self.prompt_version}")

        return with_query, with_query_and_choice
