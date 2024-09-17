import os
from typing import Union, Literal, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.enum import ModelTypes, OperatorModelTypes


class ApiSettings(BaseSettings):

    logdir: str = Field(default="logs", description="path of the logs directory")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    main_route: str = "api"
    model_type: ModelTypes = Field(default=ModelTypes.DUCKLING_OPERATOR)
    duckling_host: str = Field(default="localhost")
    duckling_port: int = Field(default=8000)
    operator_model_type: OperatorModelTypes = Field(default="bert")
    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="device to load the model on"
    )
    model_config = SettingsConfigDict(
        protected_namespaces="settings_",
        env_file=".env",
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        extra="ignore",
    )


if __name__ == "__main__":
    print(ApiSettings().model_dump_json())
