from abc import abstractmethod
from typing import Any, Tuple, List, Dict, Union
from src.misc.schemas import ProductNamedEntityExtractionSchema


class BaseModel:
    def __init__(self) -> None:
        pass

    def __call__(
        self, input_query: str, *args: Any, **kwds: Any
    ) -> ProductNamedEntityExtractionSchema:
        return self.predict(input_query=input_query)

    @abstractmethod
    def predict(self, input_query, *args: Any, **kwds: Any):
        raise NotImplementedError
