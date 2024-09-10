from typing import List, Dict, Union, Any, Tuple
import requests
import re
import json
from src.models.base import BaseModel
from src.enum import DucklingDimensionTypes, DucklingLocaleTypes

ARABIC_TEXT_PATTERN = r"[\u0600-\u06ff]|[\u0750-\u077f]|[\ufb50-\ufc3f]|[\ufe70-\ufefc]"


class DucklingHTTPModel(BaseModel):
    def __init__(
        self,
        duckling_host: str = "localhost",
        duckling_port: str = "8000",
        dim_types: List[DucklingDimensionTypes] = [
            DucklingDimensionTypes.AMOUNT_OF_MONEY
        ],
    ) -> None:
        super().__init__()
        self._duckling_url = f"http://{duckling_host}:{duckling_port}/parse"
        self._dim_types = dim_types

    def predict(
        self, input_query, *args: Any, **kwds: Any
    ) -> Tuple[bool, List[Dict[str, Union[int, str, float]]]]:
        locale_type = None
        if re.match(pattern=ARABIC_TEXT_PATTERN, string=input_query):
            locale_type = DucklingLocaleTypes.AR
        else:
            locale_type = DucklingLocaleTypes.EN
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        parsed_entities_response = requests.post(
            url=self._duckling_url,
            headers=headers,
            data=self.payload(input_query=input_query, locale=locale_type.value),
        ).json()
        if len(parsed_entities_response):
            parsed_entities = {
                "lower_range": -1,
                "upper_range": -1,
                "unit": "",
            }
            for parsed_entity in parsed_entities_response:
                if "from" in parsed_entity["value"].keys():
                    parsed_entities["lower_range"] = parsed_entity["value"]["from"][
                        "value"
                    ]
                    if (
                        parsed_entities["unit"] == ""
                        or parsed_entities["unit"] == "unknown"
                    ):
                        parsed_entities["unit"] = parsed_entity["value"]["from"]["unit"]
                if "to" in parsed_entity["value"].keys():
                    print("HERE")
                    parsed_entities["upper_range"] = parsed_entity["value"]["to"][
                        "value"
                    ]
                    if (
                        parsed_entities["unit"] == ""
                        or parsed_entities["unit"] == "unknown"
                    ):
                        parsed_entities["unit"] = parsed_entity["value"]["to"]["unit"]
            for parsed_entity in parsed_entities_response:
                if "value" in parsed_entity["value"].keys():
                    if (
                        parsed_entities["lower_range"] != -1
                        and parsed_entities["upper_range"] != -1
                    ):
                        continue
                    elif (
                        parsed_entities["lower_range"] == -1
                        and parsed_entities["upper_range"] == -1
                    ):
                        parsed_entities["lower_range"] = parsed_entity["value"]["value"]
                        parsed_entities["upper_range"] = parsed_entity["value"]["value"]
                    elif parsed_entities["lower_range"] != -1:
                        lower_range = parsed_entities["lower_range"]
                        val = parsed_entity["value"]["value"]
                        parsed_entities["upper_range"] = max(lower_range, val)
                        parsed_entities["lower_range"] = min(lower_range, val)
                    elif parsed_entities["upper_range"] != -1:
                        upper_range = parsed_entities["upper_range"]
                        val = parsed_entity["value"]["value"]
                        parsed_entities["upper_range"] = max(upper_range, val)
                        parsed_entities["lower_range"] = min(upper_range, val)
                    if (
                        parsed_entities["unit"] == ""
                        or parsed_entities["unit"] == "unknown"
                    ):
                        parsed_entities["unit"] = parsed_entity["value"]["unit"]
            if parsed_entities["unit"] == "unknown":
                parsed_entities["unit"] = ""
            return True, parsed_entities
        else:
            return False, {}

    def payload(self, input_query: str, locale: str) -> Dict[str, Union[List, str]]:

        return {
            "text": input_query,
            "locale": locale,
            "dims": json.dumps([dim_type.value for dim_type in self._dim_types]),
        }


if __name__ == "__main__":
    model = DucklingHTTPModel()
    # print(model(input_query="360"))
    # print(model(input_query="يساوي 12 ريال"))
    # print(model(input_query="من 12 ريال"))
    # print(model(input_query="أقل 12 ريال"))
    # print(model(input_query="أكثر من 12 ريال"))
    # print(model(input_query="أكثر من 12 إلى 20 ريال"))
    print(model(input_query="عطر ديور ارخص من ٢٠٠"))
