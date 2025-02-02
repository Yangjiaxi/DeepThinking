import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from tasks import get_registered_task

tasks = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "navigate",
    "penguins_in_a_table",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
]

if __name__ == "__main__":
    all_tasks = get_registered_task("all")
    for task_name, (TaskHandler, _) in all_tasks.items():
        print(f"Download: {task_name}")
        task_agent = TaskHandler("default")
        task_agent.do_load()

        print(f"\t #Test: {len(task_agent.raw_data_inference)}")
        if task_agent.raw_data_sampling:
            print(f"\t #Dev: {len(task_agent.raw_data_sampling)}")
        print("-" * 120)

    for task_name in tasks:
        task_fn, _ = get_registered_task(task_name)[task_name]
        agent = task_fn("default")
        agent.do_load()
        task_name = task_name.replace("_", "\\_")
        print(rf"\textbf{{{task_name}}}: {len(agent.raw_data_inference)}", end=", ")
