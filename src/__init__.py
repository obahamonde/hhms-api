from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from .apis import *
from .client import *
from .data import *
from .faunadb import *
from .routes import *
from .schemas import *


def create_app():
    app = FastAPI(
        title="HHMS: Hip Hop Music Studio",
        version="0.0.1",
        description="Official REST API for HHMS Application",
    )
    app.include_router(auth_app)
    app.include_router(sound_app)
    return app
