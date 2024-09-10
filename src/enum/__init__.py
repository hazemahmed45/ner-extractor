from enum import Enum


class DucklingLocaleTypes(Enum):
    EN = "en_EN"
    AR = "ar_AR"


class DucklingDimensionTypes(Enum):
    AMOUNT_OF_MONEY = "amount-of-money"
    NUMERAL = "numeral"
    ORDINAL = "ordinal"


class ModelTypes(Enum):
    DUCKLING = "duckling"
