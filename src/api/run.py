from src.api.app import create_app

# from loguru import logger
import uvicorn
from fastapi import FastAPI
from uvicorn.config import LOG_LEVELS
from src.api.settings import ApiSettings

app: FastAPI = create_app()

settings = ApiSettings()


def main():
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        loop="auto",
        log_level="info",
    )


if __name__ == "__main__":
    main()
