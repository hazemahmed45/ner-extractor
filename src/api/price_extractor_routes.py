from fastapi import APIRouter, status, Body
from fastapi.responses import JSONResponse
from loguru import logger
from src.api.settings import ApiSettings
from src.api.schema import PriceExtractionSchema
from src.models import ModelBuilder
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id


settings = ApiSettings()

price_extractor_router = APIRouter(prefix=f"/price_extractor")

model = ModelBuilder.build_model(settings=settings)


@price_extractor_router.get("/healthcheck", tags=["Healthcheck"])
async def healthcheck():
    return JSONResponse({"message": "I am alive"}, status_code=status.HTTP_200_OK)


@price_extractor_router.post("/predict", tags=["Predict"])
async def predict(
    input_query: str = Body(description="input query from the search bar"),
) -> PriceExtractionSchema:
    request_unique_id = create_unique_user_id()
    session_logger = logger.bind(user_unique_id=request_unique_id)
    session_logger.add(sink=FileHandler(user_unique_id=request_unique_id))

    session_logger.info(f"REQUEST ID -> {request_unique_id} : Request Recieved")
    session_logger.info(
        f"REQUEST ID -> {request_unique_id} : Request body is '{input_query}'"
    )
    is_detected, result = model(input_query=input_query)
    if is_detected:
        schema_result = PriceExtractionSchema(**result)
    else:
        schema_result = PriceExtractionSchema()
    session_logger.info(
        f"REQUEST ID -> {request_unique_id} : Model output is {schema_result.model_dump_json()}"
    )
    return schema_result
