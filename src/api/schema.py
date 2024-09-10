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
