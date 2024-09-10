from src.models.base import BaseModel
from src.models.duckling import DucklingHTTPModel
from src.api.settings import ApiSettings
from src.enum import ModelTypes, DucklingDimensionTypes
from src.misc.exceptions import ModelTypeError


class ModelBuilder:
    @staticmethod
    def build_model(settings: ApiSettings) -> BaseModel:
        if settings.model_type == ModelTypes.DUCKLING:
            return DucklingHTTPModel(
                duckling_host=settings.duckling_host,
                duckling_port=settings.duckling_port,
                dim_types=[
                    DucklingDimensionTypes.AMOUNT_OF_MONEY,
                ],
            )
        else:
            raise ModelTypeError("model type is not registered")
