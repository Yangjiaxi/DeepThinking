from tasks.base import BaseTaskManager


class BaseGenTaskManager(BaseTaskManager):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)
