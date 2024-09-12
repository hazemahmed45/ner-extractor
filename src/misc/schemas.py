from typing import List, Union
from pydantic import BaseModel, Field


class PriceExtractionSchema(BaseModel):
    lower_range: float = Field(
        default=-1,
        description="lower value of the price range, if set to -1, means the range is lower than the upper range",
    )
    upper_range: float = Field(
        default=-1,
        description="upper value of the price range, if set to -1, means the range is higher than the lower range",
    )
    unit: str = Field(
        default="",
        description="unit of the price, if is empty mean it was not given in the query",
    )


class SpecificationSchema(BaseModel):
    spec_name: str = Field(default="", description="Spec name")
    spec_val: str = Field(default="", description="Spec value")


class SpecsExtractionSchema(BaseModel):
    specs: List[SpecificationSchema] = Field(
        default=[], description="List of extracted specification from the query"
    )


class ProductNamedEntityExtractionSchema(BaseModel):
    super_category_extraction: Union[str, None] = Field(
        default=None, description="super category extracted name"
    )
    sub_category_extraction: Union[str, None] = Field(
        default=None, description="sub category extracted name"
    )
    brand_extraction: Union[str | None] = Field(
        default=None, description="product extracted brand name"
    )
    rate_extraction: Union[float, None] = Field(
        default=None, description="rate extracted value"
    )
    price_extraction: Union[PriceExtractionSchema, None] = Field(
        default=None, description="price extraction range"
    )
    specs_extraction: Union[SpecsExtractionSchema, None] = Field(
        default=None, description="All product specifiction extracted"
    )
