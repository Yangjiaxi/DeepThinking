from tasks import register_task, TaskType
from tasks.base_selection import BaseProbTaskManager
import re

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class MmluBaseProbTask(BaseProbTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.CHOICES = 4
        self.can_be_stratified = False
        self.num_base_shot = 1

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append(
                {
                    "query": e["question"],
                    "choices": e["choices"],
                    "answer_idx": e["answer"],
                }
            )
        return data

    def promptify_input(self, query):
        with_query = f"Question: {query}\nAnswer:"
        return with_query

    def promptify_input_with_choice(self, query, choice):
        with_query_and_choice = f"{self.promptify_input(query)} {choice}"
        return with_query_and_choice


def make_signature(subject_name):
    def fn(self):
        return {
            "inference": ("cais/mmlu", subject_name, "test"),
            "sampling": ("cais/mmlu", subject_name, "dev"),
        }

    return fn


name_to_classes = {}

for subject in SUBJECTS:
    name_to_classes[subject] = type(f"MMLU{subject}ProbTask", (MmluBaseProbTask,), {"dataset_signature": make_signature(subject)})
    register_task(name=subject, task_type=TaskType.Selection, suite=["mmlu", "multiple-choice"])(name_to_classes[subject])
