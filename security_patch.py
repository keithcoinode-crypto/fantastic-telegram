import os
from fastapi import FastAPI

def secure_app(APP_NAME, APP_VERSION):
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    return app

def get_credentials():
    username = os.getenv("OWNER_USERNAME", "keith")
    password = os.getenv("OWNER_PASSWORD", "omega001")
    return username, password
