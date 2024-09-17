from pathlib import Path
from fastapi import FastAPI, APIRouter
from src.misc.settings import ApiSettings
from src.api.extractor_routes import price_extractor_router


# def get_active_branch_name():

#     head_dir = Path(".") / ".git" / "HEAD"
#     with head_dir.open("r") as f:
#         content = f.read().splitlines()

#     for line in content:
#         if line[0:4] == "ref:":
#             return line.partition("refs/heads/")[2]


def create_api_desc() -> str:

    desc = f"""Deployed API for Named Entity Recognition"""
    return desc


def create_app() -> FastAPI:
    settings = ApiSettings()
    api_desc = create_api_desc()
    app: FastAPI = FastAPI(
        title="Named Entity Recognition API",
        summary=api_desc,
        # description=create_api_desc(),
        version="v1",
        contact={"name": "Hazem A.Haroun", "email": "hazemahmed45@gmail.com"},
        docs_url=f"/{settings.main_route}/docs",
        redoc_url=f"/{settings.main_route}/redoc",
    )
    api_router = APIRouter(prefix=f"/{settings.main_route}")

    api_router.include_router(router=price_extractor_router)

    app.include_router(api_router)

    return app
