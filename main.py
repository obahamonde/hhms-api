from dotenv import load_dotenv

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from src import *

app = create_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.on_event("startup")
async def startup_event():
    await FaunaModel.create_all()
