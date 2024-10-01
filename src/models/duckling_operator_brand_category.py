from abc import abstractmethod
from typing import List, Dict, Union, Any
import os
import requests
import re
import json
import fasttext
import torch
from fasttext.FastText import _FastText
from transformers import BertTokenizer
from src.models.base import BaseModel
from src.models.bert_cls import BERTClassifier
from src.enum import (
    DucklingDimensionTypes,
    DucklingLocaleTypes,
    OperatorClassTypes,
)
from src.misc.schemas import PriceExtractionSchema, ProductNamedEntityExtractionSchema

ARABIC_TEXT_PATTERN = r"[\u0600-\u06ff]|[\u0750-\u077f]|[\ufb50-\ufc3f]|[\ufe70-\ufefc]"


class BaseClassifierModel(BaseModel):
    def __init__(self, model_ckpt_path: str) -> None:
        super().__init__()
        self._model = self.load_model(model_ckpt_path)

    def __call__(self, input_query: str, *args: Any, **kwds: Any) -> OperatorClassTypes:
        return super().__call__(input_query, *args, **kwds)

    @abstractmethod
    def predict(self, input_query, *args: Any, **kwds: Any) -> str:
        raise NotImplementedError()

    @abstractmethod
    def load_model(self, model_ckpt_path):
        raise NotImplementedError()


class BertClassifierModel(BaseClassifierModel):
    def __init__(
        self,
        model_ckpt_path: str = os.path.join("ckpt", "bert_brands"),
    ) -> None:
        model_config: Dict = json.load(
            open(os.path.join(model_ckpt_path, "model_config.json"), "r")
        )
        self._index2class: List[str] = model_config["index2class"]
        self._bert_model_name = model_config["bert_model_name"]
        self._tokenizer = self.load_tokenizer(bert_model_name=self._bert_model_name)
        self._num_classes = len(self._index2class)
        self._device = model_config["device"]
        self._max_length = model_config["max_length"]
        super().__init__(model_ckpt_path)

    def load_model(self, model_ckpt_path) -> BERTClassifier:
        model = BERTClassifier(self._bert_model_name, self._num_classes).to(
            self._device
        )
        model.eval()
        model.load_state_dict(
            torch.load(
                os.path.join(
                    model_ckpt_path,
                    "bert_classifier.pth",
                ),
                map_location=self._device,
            )
        )
        return model

    def load_tokenizer(self, bert_model_name: str) -> BertTokenizer:
        return BertTokenizer.from_pretrained(bert_model_name)

    def predict(self, input_query: str, *args: Any, **kwds: Any) -> str:
        encoding = self._tokenizer(
            input_query,
            return_tensors="pt",
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            _, pred_class = torch.max(outputs, dim=1)
            pred_class = self._index2class[pred_class.item()]
            return pred_class


class BertOperatorModel(BertClassifierModel):
    def __init__(
        self,
        model_ckpt_path: str = os.path.join("ckpt", "bert_operator"),
    ) -> None:
        super().__init__(model_ckpt_path)

    def predict(self, input_query, *args: Any, **kwds: Any) -> OperatorClassTypes:
        pred_class = super().predict(input_query, *args, **kwds)
        for class_type in OperatorClassTypes:
            if class_type.value == pred_class:
                pred_class = class_type
        return pred_class


class DucklingHTTPBertNERModel(BaseModel):
    def __init__(
        self,
        duckling_host: str = "localhost",
        duckling_port: str = "8000",
        dim_types: List[DucklingDimensionTypes] = [
            DucklingDimensionTypes.AMOUNT_OF_MONEY,
            DucklingDimensionTypes.NUMERAL,
            DucklingDimensionTypes.ORDINAL,
        ],
        category_model: BertClassifierModel = None,
        brand_model: BertClassifierModel = None,
        operator_model: BertOperatorModel = None,
    ) -> None:
        super().__init__()
        self._duckling_url = f"http://{duckling_host}:{duckling_port}/parse"
        self._dim_types = dim_types
        self.operator_model = operator_model
        self.currency_pattern_map = {
            "SAR": r"\b(ريال سعودي|ر\.س|ريال|SAR|saudi riyal|riyal|sr)\b",
            "$": r"\b(دولار امريكي|دولار أمريكي|دولار أمريكى|دولار امريكى|دولار|دولارًا|\$|USD|united states dollar|dollar)\b",
            "AED": r"\b(درهم اماراتي|درهم إماراتي|درهم اماراتى|درهم إماراتى|درهم|د\.إ|AED|emirates dirham|emirate dirham|dirham)\b",
            "EGP": r"\b(جنيه مصري|جنيه مصرى|جنيه|ج\.م|£|egyptian pound|pound|le|LE|EGP)\b",
        }
        self.category_model = category_model
        self.brand_model = brand_model

    def predict(
        self, input_query, *args: Any, **kwds: Any
    ) -> ProductNamedEntityExtractionSchema:
        input_query = input_query.replace("اغلي", "أغلى").replace("أغلي", "أغلى")
        extraction_result = ProductNamedEntityExtractionSchema()
        price_extraction_result = PriceExtractionSchema()
        extraction_result.sub_category_extraction = input_query
        extraction_result.price_extraction = price_extraction_result
        operator_class_pred: OperatorClassTypes = self.operator_model(
            input_query=input_query
        )
        brand_class_pred: str = self.brand_model(input_query=input_query)
        category_class_pred: str = self.category_model(input_query=input_query)
        super_cat, sub_cat = (
            category_class_pred.split("%")[0],
            category_class_pred.split("%")[1],
        )
        extraction_result.brand_extraction = (
            None if brand_class_pred == "NONE" else brand_class_pred
        )

        extraction_result.super_category_extraction = (
            None if super_cat == "NONE" else super_cat
        )
        extraction_result.sub_category_extraction = (
            None if sub_cat == "NONE" else sub_cat
        )

        locale_type = None
        if re.match(pattern=ARABIC_TEXT_PATTERN, string=input_query):
            locale_type = DucklingLocaleTypes.AR
        else:
            locale_type = DucklingLocaleTypes.EN
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if operator_class_pred == OperatorClassTypes.NONE:
            return extraction_result

        parsed_entities_response = requests.post(
            url=self._duckling_url,
            headers=headers,
            data=self.payload(input_query=input_query, locale=locale_type.value),
        ).json()
        if len(parsed_entities_response):
            for parsed_entity in parsed_entities_response:
                if "from" in parsed_entity["value"].keys():
                    price_extraction_result.lower_range = parsed_entity["value"][
                        "from"
                    ]["value"]

                    if (
                        price_extraction_result.unit == ""
                        or price_extraction_result.unit == "unknown"
                    ):
                        price_extraction_result.unit = parsed_entity["value"]["from"][
                            "unit"
                        ]
                if "to" in parsed_entity["value"].keys():
                    price_extraction_result.upper_range = parsed_entity["value"]["to"][
                        "value"
                    ]
                    if (
                        price_extraction_result.unit == ""
                        or price_extraction_result.unit == "unknown"
                    ):
                        price_extraction_result.unit = parsed_entity["value"]["to"][
                            "unit"
                        ]
            for parsed_entity in parsed_entities_response:
                if "value" in parsed_entity["value"].keys():
                    if (
                        price_extraction_result.lower_range != -1
                        and price_extraction_result.upper_range != -1
                    ):
                        continue
                    elif (
                        price_extraction_result.lower_range == -1
                        and price_extraction_result.upper_range == -1
                    ):
                        if operator_class_pred == OperatorClassTypes.LE:
                            price_extraction_result.upper_range = parsed_entity[
                                "value"
                            ]["value"]
                        elif operator_class_pred == OperatorClassTypes.EQ:
                            price_extraction_result.lower_range = parsed_entity[
                                "value"
                            ]["value"]
                            price_extraction_result.upper_range = parsed_entity[
                                "value"
                            ]["value"]
                        elif operator_class_pred == OperatorClassTypes.GE:
                            price_extraction_result.lower_range = parsed_entity[
                                "value"
                            ]["value"]
                        elif operator_class_pred == OperatorClassTypes.GELE:
                            numbers_pattern = r"[+-]?\d+\.\d+|\d+"
                            digits_matches = re.findall(numbers_pattern, input_query)
                            if len(digits_matches):
                                digit_matched = None
                                for digit in digits_matches:
                                    if float(digit) != float(
                                        parsed_entity["value"]["value"]
                                    ):

                                        digit_matched = float(digit)
                                        break
                                if digit_matched:
                                    price_extraction_result.lower_range = min(
                                        parsed_entity["value"]["value"], digit_matched
                                    )
                                    price_extraction_result.upper_range = max(
                                        parsed_entity["value"]["value"], digit_matched
                                    )
                                else:
                                    price_extraction_result.lower_range = parsed_entity[
                                        "value"
                                    ]["value"]
                                    price_extraction_result.upper_range = parsed_entity[
                                        "value"
                                    ]["value"]
                            else:
                                price_extraction_result.lower_range = parsed_entity[
                                    "value"
                                ]["value"]
                                price_extraction_result.upper_range = parsed_entity[
                                    "value"
                                ]["value"]

                    elif price_extraction_result.lower_range != -1:
                        lower_range = price_extraction_result.lower_range
                        val = parsed_entity["value"]["value"]
                        if price_extraction_result.upper_range == -1:
                            if operator_class_pred == OperatorClassTypes.EQ:
                                price_extraction_result.upper_range = val
                            elif operator_class_pred == OperatorClassTypes.LE:
                                price_extraction_result.upper_range = lower_range
                                price_extraction_result.lower_range = -1
                            elif operator_class_pred == OperatorClassTypes.GELE:
                                price_extraction_result.upper_range = max(
                                    lower_range, val
                                )
                                price_extraction_result.lower_range = min(
                                    lower_range, val
                                )
                        else:
                            price_extraction_result.upper_range = max(lower_range, val)
                            price_extraction_result.lower_range = min(lower_range, val)
                    elif price_extraction_result.upper_range != -1:
                        upper_range = price_extraction_result.upper_range
                        val = parsed_entity["value"]["value"]
                        if price_extraction_result.lower_range == -1:
                            if operator_class_pred == OperatorClassTypes.EQ:
                                price_extraction_result.lower_range = val
                            elif operator_class_pred == OperatorClassTypes.LE:
                                price_extraction_result.lower_range = upper_range
                                price_extraction_result.upper_range = -1
                            elif operator_class_pred == OperatorClassTypes.GELE:
                                price_extraction_result.upper_range = max(
                                    upper_range, val
                                )
                                price_extraction_result.lower_range = min(
                                    upper_range, val
                                )
                        else:
                            price_extraction_result.upper_range = max(upper_range, val)
                            price_extraction_result.lower_range = min(upper_range, val)

                    if (
                        price_extraction_result.unit == ""
                        or price_extraction_result.unit == "unknown"
                    ):
                        price_extraction_result.unit = parsed_entity["value"]["unit"]
            if price_extraction_result.unit == "unknown":
                price_extraction_result.unit = ""
        else:
            numbers_pattern = r"[+-]?\d+\.\d+|\d+"
            digits_matches = [
                float(digit) for digit in re.findall(numbers_pattern, input_query)
            ]
            if len(digits_matches):
                if operator_class_pred == OperatorClassTypes.LE:
                    price_extraction_result.upper_range = max(digits_matches)
                elif operator_class_pred == OperatorClassTypes.EQ:
                    price_extraction_result.upper_range = digits_matches[0]
                    price_extraction_result.lower_range = digits_matches[0]
                elif operator_class_pred == OperatorClassTypes.GE:
                    price_extraction_result.lower_range = min(digits_matches)
                elif operator_class_pred == OperatorClassTypes.GELE:
                    price_extraction_result.upper_range = max(digits_matches)
                    price_extraction_result.lower_range = min(digits_matches)

            for currency, cur_pattern in self.currency_pattern_map.items():

                currency_matches = re.findall(cur_pattern, input_query, re.IGNORECASE)
                if len(currency_matches):
                    price_extraction_result.unit = currency
                    break
        for currency, cur_pattern in self.currency_pattern_map.items():
            currency_matches = re.findall(
                cur_pattern, price_extraction_result.unit, re.IGNORECASE
            )
            if len(currency_matches):
                price_extraction_result.unit = currency

                break
        return extraction_result

    def payload(self, input_query: str, locale: str) -> Dict[str, Union[List, str]]:

        return {
            "text": input_query,
            "locale": locale,
            "dims": json.dumps([dim_type.value for dim_type in self._dim_types]),
        }


if __name__ == "__main__":
    model = DucklingHTTPBertNERModel()
    # print(model(input_query="360"))
    # print(model(input_query="يساوي 12 ريال").model_dump())
    # # print(model(input_query="من 12 ريال"))
    # # print(model(input_query="أقل 12 ريال"))
    # # print(model(input_query="أكثر من 12 ريال"))
    # print(model(input_query="أكثر من 12 إلى 20 ريال"))
    # # print(model(input_query="عطر ديور ارخص من ٢٠٠").model_dump())
    # print(model(input_query="اقل من 200 ريال").model_dump())
    # print(model(input_query="أرخص من 200").model_dump())
    # print(model(input_query="cheaper than 200 ").model_dump())
    # pattern = r"\b(ريال|دولار|درهم|دينار|يورو|جنيه|ليرة|د\.إ|ريال سعودي|ر\.س|جنيه مصري|جنيه مصرى|دولار امريكي|دولار امريكى|درهم اماراتي|درهم إماراتي|درهم اماراتى|درهم إماراتى|ج\.م|\$|€|£|USD|AED|SAR|EUR|GBP|usd|aed|sar|eur|saudi riyal|riyal|sr|egyptian pound|pound|le|LE|emirates dirham|dirham|emirate dirham|dollar|united states dollar)\b"
    pattern = r"\b(ريال سعودي|ر\.س|جنيه مصري|جنيه مصرى|دولار امريكي|دولار أمريكي|دولار أمريكى|دولار امريكى|درهم اماراتي|درهم إماراتي|درهم اماراتى|درهم إماراتى|ريال|دولار|درهم|دينار|يورو|جنيه|ليرة|د\.إ|ج\.م|\$|€|£|USD|AED|SAR|EUR|GBP|usd|aed|sar|eur|saudi riyal|egyptian pound|riyal|sr|pound|le|LE|emirates dirham|emirate dirham|dirham|united states dollar|dollar)\b"

    # Example text containing different currencies in Arabic and English
    text = """
    سعر المنتج هو 120 ريال سعودي.
    التكلفة الكلية 50 دولار أمريكي.
    The laptop costs $500 USD.
    سعر الهاتف هو 250 درهم إماراتي.
    سعر الساعة 99.99€.
    """

    # Find all matches of currencies in the text
    matches = re.findall(pattern, text, re.IGNORECASE)
    print(matches)
