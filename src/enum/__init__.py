from enum import Enum


class DucklingLocaleTypes(Enum):
    EN = "en_EN"
    AR = "ar_AR"


class DucklingDimensionTypes(Enum):
    AMOUNT_OF_MONEY = "amount-of-money"
    NUMERAL = "numeral"
    ORDINAL = "ordinal"
    VOLUME = "volume"
    QUANTITY = "quantity"


class ModelTypes(Enum):
    DUCKLING = "duckling"
    DUCKLING_OPERATOR = "duckling_operator"


class FastTextOperatorClassTypes(Enum):
    EQ = "__label__EQ"
    LE = "__label__LE"
    GE = "__label__GE"
    GELE = "__label__GELE"
    NONE = "__label__NONE"
