import pkgutil
from importlib import import_module
from enum import Enum, auto
from typing import Iterable
import os

TASK_REGISTRY = {}  # `task_name`: task_fn
SUITE_REGISTREY = {}  # `suite_name`: {`task_name`: task_fn}


class TaskType(Enum):
    Generation = auto()
    Selection = auto()


def list_all_tasks():
    return [*TASK_REGISTRY.keys(), *SUITE_REGISTREY.keys(), "all"]


def register_task(name, task_type: TaskType, suite=None):
    def decorate(fn):
        if name == "all" or suite == "all":
            raise ValueError("`all` is a preserved name, pick another!")
        task_packed = (fn, task_type)
        the_suite = suite
        if the_suite is not None:
            if isinstance(the_suite, str):
                the_suite = [the_suite]

            if isinstance(the_suite, Iterable):
                for suite_item in the_suite:
                    if suite_item not in SUITE_REGISTREY:
                        SUITE_REGISTREY[suite_item] = {}
                    SUITE_REGISTREY[suite_item][name] = task_packed

        TASK_REGISTRY[name] = task_packed
        return fn

    return decorate


def get_registered_task(name):
    if name in TASK_REGISTRY:
        task_packed = TASK_REGISTRY[name]
        return {name: task_packed}

    if name in SUITE_REGISTREY:
        return SUITE_REGISTREY[name]

    if name == "all":
        return TASK_REGISTRY
    raise ValueError(f"{name}: neither a registered task nor a suite!")


task_folders = [
    "tasks/tasks_generation",
    "tasks/tasks_selection",
]

for task_folder in task_folders:
    for _, module, _ in pkgutil.iter_modules([task_folder]):
        build_path = f"{task_folder.replace('/', '.')}.{module}"
        import_module(build_path)

rank = int(os.environ.get("RANK", 0))

if rank == 0:
    task_str = ", ".join(TASK_REGISTRY.keys())

    print(f"[REGISTRY] Tasks: {task_str}")
    for k, vs in SUITE_REGISTREY.items():
        sub = ", ".join(vs.keys())
        print(f"[REGISTRY] Suites [{k}]: {sub}")
