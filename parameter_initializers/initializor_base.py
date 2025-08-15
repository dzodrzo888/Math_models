from abc import ABC, abstractmethod

class InitializerBaseModel(ABC):

    @abstractmethod
    def initialize_parameters(self, n_features: int):
        pass
