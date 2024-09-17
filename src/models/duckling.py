from typing import List, Dict, Union, Any, Tuple
import requests
import re
import json
from src.models.base import BaseModel
from src.enum import DucklingDimensionTypes, DucklingLocaleTypes
from src.misc.schemas import PriceExtractionSchema, ProductNamedEntityExtractionSchema


ARABIC_TEXT_PATTERN = r"[\u0600-\u06ff]|[\u0750-\u077f]|[\ufb50-\ufc3f]|[\ufe70-\ufefc]"


class DucklingHTTPModel(BaseModel):
    def __init__(
        self,
        duckling_host: str = "localhost",
        duckling_port: str = "8000",
        dim_types: List[DucklingDimensionTypes] = [
            DucklingDimensionTypes.AMOUNT_OF_MONEY,
            DucklingDimensionTypes.NUMERAL,
            DucklingDimensionTypes.ORDINAL,
        ],
    ) -> None:
        super().__init__()
        self._duckling_url = f"http://{duckling_host}:{duckling_port}/parse"
        self._dim_types = dim_types
        self.currency_pattern_map = {
            "SAR": r"\b(ريال سعودي|ر\.س|ريال|SAR|saudi riyal|riyal|sr)\b",
            "$": r"\b(دولار امريكي|دولار أمريكي|دولار أمريكى|دولار امريكى|دولار|دولارًا|\$|USD|united states dollar|dollar)\b",
            "AED": r"\b(درهم اماراتي|درهم إماراتي|درهم اماراتى|درهم إماراتى|درهم|د\.إ|AED|emirates dirham|emirate dirham|dirham)\b",
            "EGP": r"\b(جنيه مصري|جنيه مصرى|جنيه|ج\.م|£|egyptian pound|pound|le|LE|EGP)\b",
        }

    def predict(
        self, input_query, *args: Any, **kwds: Any
    ) -> ProductNamedEntityExtractionSchema:
        extraction_results = ProductNamedEntityExtractionSchema()

        if re.match(pattern=ARABIC_TEXT_PATTERN, string=input_query):
            locale_type = DucklingLocaleTypes.AR
        else:
            locale_type = DucklingLocaleTypes.EN
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        parsed_entities_response = requests.post(
            url=self._duckling_url,
            headers=headers,
            data=self.payload(input_query=input_query, locale=locale_type.value),
            timeout=1000,
        ).json()
        if len(parsed_entities_response):
            extraction_results.sub_category_extraction = input_query
            price_extraction_result: PriceExtractionSchema = PriceExtractionSchema()
            extraction_results.price_extraction = price_extraction_result

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
                extraction_results.sub_category_extraction = (
                    extraction_results.sub_category_extraction.replace(
                        parsed_entity["body"], ""
                    )
                )
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
                        price_extraction_result.lower_range = parsed_entity["value"][
                            "value"
                        ]
                        price_extraction_result.upper_range = parsed_entity["value"][
                            "value"
                        ]
                    elif price_extraction_result.lower_range != -1:
                        lower_range = price_extraction_result.lower_range
                        val = parsed_entity["value"]["value"]
                        price_extraction_result.upper_range = max(lower_range, val)
                        price_extraction_result.lower_range = min(lower_range, val)
                    elif price_extraction_result.upper_range != -1:
                        upper_range = price_extraction_result.upper_range
                        val = parsed_entity["value"]["value"]
                        price_extraction_result.upper_range = max(upper_range, val)
                        price_extraction_result.lower_range = min(upper_range, val)
                    if (
                        price_extraction_result.unit == ""
                        or price_extraction_result.unit == "unknown"
                    ):
                        price_extraction_result.unit = parsed_entity["value"]["unit"]
                extraction_results.sub_category_extraction = (
                    extraction_results.sub_category_extraction.replace(
                        parsed_entity["body"], ""
                    )
                )
            if price_extraction_result.unit == "unknown":
                price_extraction_result.unit = ""
            extraction_results.sub_category_extraction = " ".join(
                extraction_results.sub_category_extraction.split()
            )
            for currency, cur_pattern in self.currency_pattern_map.items():
                currency_matches = re.findall(
                    cur_pattern, price_extraction_result.unit, re.IGNORECASE
                )
                if len(currency_matches):
                    price_extraction_result.unit = currency

        return extraction_results

    def payload(self, input_query: str, locale: str) -> Dict[str, Union[List, str]]:

        return {
            "text": input_query,
            "locale": locale,
            "dims": json.dumps([dim_type.value for dim_type in self._dim_types]),
        }


if __name__ == "__main__":
    model = DucklingHTTPModel()
    # print(model(input_query="360"))

    print(
        json.dumps(
            model(input_query="يساوي 12 ريال و عشرون ريال").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="من 12 ريال").model_dump(), indent=3, ensure_ascii=False
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="أقل 12 ريال").model_dump(), indent=3, ensure_ascii=False
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="أكثر من 12 ريال").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="أكثر من 12 إلى 20 ريال").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="عطر ديور ارخص من ٢٠٠").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="عطر ديور ارخص من ٢٠٠").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
