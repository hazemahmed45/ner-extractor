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
    DUCKLING_BERT_NER = "duckling_bert_ner"
    T5_NER = "t5_ner"


class OperatorModelTypes(Enum):
    FASTTEXT = "fasttext"
    BERT = "bert"


class OperatorClassTypes(Enum):
    EQ = "__label__EQ"
    LE = "__label__LE"
    GE = "__label__GE"
    GELE = "__label__GELE"
    NONE = "__label__NONE"


class T5DomainClassTypes(Enum):
    RATE = "[RATE]"
    SPECS = "[SPECS]"
    BRAND = "[BRAND]"
    SUBCATEGORY = "[SUBCATEGORY]"
    SUPERCATEGORY = "[SUPERCATEGORY]"
    PRICE = "[PRICE]"


class T5PriceSubclassTypes(Enum):
    EQ = "EQ"
    LE = "LE"
    GE = "GE"
    CURRENCY = "CURRENCY"
