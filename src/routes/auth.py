from os import environ

from aiohttp import ClientSession
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from ..schemas import FreeSoundUser

FREESOUND_CLIENT_ID = environ["FREESOUND_CLIENT_ID"]
FREESOUND_CLIENT_SECRET = environ["FREESOUND_CLIENT_SECRET"]
FREESOUND_AUTH_URL = "https://freesound.org/apiv2/oauth2/authorize/"
CLIENT_URL = "http://localhost:3000/"  # "https://hhms.vercel.app"

app = APIRouter(prefix="/api", tags=["auth"])


@app.get("/")
async def authorize_endpoint():
    """Auth0 Callback"""
    return RedirectResponse(
        f"{FREESOUND_AUTH_URL}?client_id={FREESOUND_CLIENT_ID}&response_type=code"
    )


@app.get("/callback")
async def callback_endpoint(code: str):
    """Auth0 Callback"""
    async with ClientSession() as session:
        response = await session.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            data={
                "client_id": FREESOUND_CLIENT_ID,
                "client_secret": FREESOUND_CLIENT_SECRET,
                "grant_type": "authorization_code",
                "code": code,
            },
        )
        data = await response.json()
        return RedirectResponse(
            f"{CLIENT_URL}?token={data['access_token']}&refresh_token={data['refresh_token']}"
        )


@app.get("/user")
async def user_endpoint(token: str):
    """Auth0 Callback"""
    async with ClientSession() as session:
        response = await session.get(
            "https://freesound.org/apiv2/me/",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = await response.json()
        return await FreeSoundUser(**data).save()
