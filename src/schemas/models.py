from pydantic import BaseModel, HttpUrl

from ..data import Document, FaunaModel, Field


class Avatar(Document):
    small: str
    medium: str
    large: str


class FreeSoundUser(FaunaModel):
    url: str
    username: str = Field(..., index=True)
    about: str
    home_page: str
    avatar: Avatar
    date_joined: str  # Consider using datetime if you control the format
    num_sounds: int = Field(..., index=True)
    sounds: str
    num_packs: int
    packs: str
    num_posts: int
    num_comments: int
    bookmark_categories: str
    email: str = Field(..., unique=True)
    unique_id: int = Field(..., unique=True)


class AudioFile(BaseModel):
    user: str
    key: str
    vector:list[float]

class AudioUpload(FaunaModel):
    user:str = Field(...,index=True)
    key:str = Field(...,index=True)