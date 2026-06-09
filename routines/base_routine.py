from abc import ABC, abstractmethod


class Routine(ABC):

    @abstractmethod
    def run(self):
        pass
