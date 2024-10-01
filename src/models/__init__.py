import os
from src.models.base import BaseModel
from src.models.duckling import DucklingHTTPModel
from src.models.t5_ner_extractor import T5NERModel
from src.models.duckling_operator import (
    DucklingHTTPOperatorModel,
    BaseOperatorModel,
    BertOperatorModel,
    FasttextOperatorModel,
)
from src.models.duckling_operator_brand_category import (
    DucklingHTTPBertNERModel,
    BertClassifierModel,
    BertOperatorModel as BertNEROperatorModel,
)
from src.misc.settings import ApiSettings
from src.enum import ModelTypes, DucklingDimensionTypes, OperatorModelTypes
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
        elif settings.model_type == ModelTypes.T5_NER:
            return T5NERModel(
                model_ckpt_dir=os.path.join("ckpt", "ner_model"),
                device=settings.device,
            )
        elif settings.model_type == ModelTypes.DUCKLING_OPERATOR:
            operator_model: BaseOperatorModel | None = None
            if settings.operator_model_type == OperatorModelTypes.BERT:
                operator_model = BertOperatorModel()
            elif settings.operator_model_type == OperatorModelTypes.FASTTEXT:
                operator_model = FasttextOperatorModel()
            return DucklingHTTPOperatorModel(
                duckling_host=settings.duckling_host,
                duckling_port=settings.duckling_port,
                dim_types=[
                    DucklingDimensionTypes.AMOUNT_OF_MONEY,
                ],
                operator_model=operator_model,
            )
        elif settings.model_type == ModelTypes.DUCKLING_BERT_NER:
            return DucklingHTTPBertNERModel(
                duckling_host=settings.duckling_host,
                duckling_port=settings.duckling_port,
                dim_types=[
                    DucklingDimensionTypes.AMOUNT_OF_MONEY,
                ],
                operator_model=BertNEROperatorModel(
                    model_ckpt_path=os.path.join("ckpt", "bert_operator")
                ),
                brand_model=BertClassifierModel(
                    model_ckpt_path=os.path.join("ckpt", "bert_brands")
                ),
                category_model=BertClassifierModel(
                    model_ckpt_path=os.path.join("ckpt", "bert_category")
                ),
            )
        else:
            raise ModelTypeError("model type is not registered")
