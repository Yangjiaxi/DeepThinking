from tasks.base_selection import BaseProbTaskManager
from tasks import register_task, TaskType


@register_task(
    name="arc_e",
    task_type=TaskType.Selection,
    suite=["arc", "multiple-choice"],
)
class ArcEasyProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

        self.expected_num_choices = 4

    def dataset_signature(self):
        return {
            "inference": ("ai2_arc", "ARC-Easy", "test"),
            "sampling": ("ai2_arc", "ARC-Easy", "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = e["question"]
            choices = e["choices"]["text"]
            if len(choices) != self.expected_num_choices:
                continue
            if e["answerKey"] in "ABCD":
                label = ord(e["answerKey"]) - ord("A")
            elif e["answerKey"] in "1234":
                label = ord(e["answerKey"]) - ord("1")
            else:
                continue
            data.append({"query": query, "choices": choices, "answer_idx": label})

        return data

    def promptify_input(self, query):
        with_query = f"Question: {query}\nAnswer:"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice


@register_task(
    name="arc_c",
    task_type=TaskType.Selection,
    suite=["arc", "multiple-choice"],
)
class ArcChallengeProbTask(ArcEasyProbTask):
    def dataset_signature(self):
        return {
            "inference": ("ai2_arc", "ARC-Challenge", "test"),
            "sampling": ("ai2_arc", "ARC-Challenge", "train"),
        }
