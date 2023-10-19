from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()
from typing import List, Literal, Optional

from pydantic import (BaseModel, Field,  # pylint: disable=no-name-in-module
                      HttpUrl)

from ..client import *

Sort = Literal[
    "score",
    "duration_desc",
    "duration_asc",
    "created_desc",
    "created_asc",
    "downloads_desc",
    "downloads_asc",
    "rating_desc",
    "rating_asc",
]

Filter = Literal["bass", "melody", "reverb"]


class QueryFilter(BaseModel):
    id: Optional[int]
    username: Optional[str]
    created: Optional[str]
    original_filename: Optional[str]
    description: Optional[str]
    tag: Optional[List[str]]
    license: Optional[str]
    is_remix: Optional[bool]
    was_remixed: Optional[bool]
    pack: Optional[str]
    is_geotagged: Optional[bool]
    type: Optional[str]
    duration: Optional[float]  ##############
    bitdepth: Optional[int]
    bitrate: Optional[float]
    samplerate: Optional[int]
    filesize: Optional[int]
    channels: Optional[int]
    md5: Optional[str]
    num_downloads: Optional[int]
    avg_rating: Optional[float]
    num_ratings: Optional[int]
    comment: Optional[str]
    comments: Optional[int]
    ac_loudness: Optional[float]
    ac_dynamic_range: Optional[float]
    ac_temporal_centroid: Optional[float]
    ac_log_attack_time: Optional[float]
    ac_single_event: Optional[bool]
    ac_tonality: Optional[str]
    ac_tonality_confidence: Optional[float]
    ac_loop: Optional[bool]
    ac_tempo: Optional[int]  #################
    ac_tempo_confidence: Optional[float]
    ac_note_midi: Optional[int]  #################
    ac_note_name: Optional[str]
    ac_note_frequency: Optional[float]
    ac_note_confidence: Optional[float]
    ac_brightness: Optional[float]
    ac_depth: Optional[float]
    ac_hardness: Optional[float]
    ac_roughness: Optional[float]
    ac_boominess: Optional[float]
    ac_warmth: Optional[float]
    ac_sharpness: Optional[float]
    ac_reverb: Optional[bool]

    def __call__(self) -> str:
        filters: List[str] = []
        for field in self.__fields__.values():
            if self.dict().get(field.name) is not None:
                filters.append(f"{field.name}:{self.dict()[field.name]}")
            timelapse = field.field_info.extra.get("Timelapse")
            if timelapse:
                start, end = timelapse
                filters.append(f"{field.name}:[{start} TO {end}]")

            range_ = field.field_info.extra.get("Range")
            if range_:
                start, end = range_
                filters.append(f"{field.name}:[{start} TO {end}]")

            starts_at = field.field_info.extra.get("From")
            if starts_at:
                filters.append(f"{field.name}:[{starts_at} TO *]")

            ends_at = field.field_info.extra.get("To")
            if ends_at:
                filters.append(f"{field.name}:[* TO {ends_at}]")

            or_ = field.field_info.extra.get("Or")
            if or_:
                filters.append(f"{field.name}:( {' OR '.join(or_)} )")

            and_ = field.field_info.extra.get("And")
            if and_:
                filters.append(f"{field.name}:( {' AND '.join(and_)} )")

        if not filters:
            raise ValueError("No fields to filter")

        return " ".join(filters)


class BassFilter(QueryFilter):
    duration: Optional[float] = Field(None, Range=(5, 30))
    ac_tempo: Optional[int] = Field(None, Range=(90, 150))
    ac_note_midi: Optional[int] = Field(None, Range=(28, 55))


class MelodyFilter(QueryFilter):
    duration: Optional[float] = Field(None, Range=(5, 30))
    ac_tempo: Optional[int] = Field(None, Range=(90, 150))
    ac_note_midi: Optional[int] = Field(None, Range=(36, 101))


class ReverbFilter(QueryFilter):
    duration: Optional[float] = Field(None, Range=(5, 30))
    ac_tempo: Optional[int] = Field(None, Range=(90, 150))
    ac_note_midi: Optional[int] = Field(None, Range=(28, 101))
    ac_reverb: Optional[bool] = Field()


FilterMapping = {
    "bass": BassFilter(),
    "melody": MelodyFilter(),
    "reverb": ReverbFilter(),
}


class SearchQuery(BaseModel):
    query: str
    filter: Optional[str]
    sort: Optional[Sort]


class SearchResult(BaseModel):
    id: int
    name: str
    tags: List[str]
    license: HttpUrl
    username: str


class SearchResponse(BaseModel):
    count: int
    previous: Optional[HttpUrl]
    next: Optional[HttpUrl]
    results: List[SearchResult]


class Previews(BaseModel):
    preview_hq_mp3: Optional[HttpUrl]
    preview_hq_ogg: Optional[HttpUrl]
    preview_lq_mp3: Optional[HttpUrl]
    preview_lq_ogg: Optional[HttpUrl]


class Images(BaseModel):
    waveform_m: Optional[HttpUrl]
    waveform_l: Optional[HttpUrl]
    spectral_m: Optional[HttpUrl]
    spectral_l: Optional[HttpUrl]
    waveform_bw_m: Optional[HttpUrl]
    waveform_bw_l: Optional[HttpUrl]
    spectral_bw_m: Optional[HttpUrl]
    spectral_bw_l: Optional[HttpUrl]


class SoundInstance(BaseModel):
    id: int
    url: HttpUrl
    name: str
    tags: List[str]
    description: str
    geotag: Optional[str]
    created: str
    license: HttpUrl
    type: str
    channels: int
    filesize: int
    bitrate: int
    bitdepth: int
    duration: float
    samplerate: float
    username: str
    pack: Optional[str]
    pack_name: Optional[str]
    download: HttpUrl
    bookmark: Optional[HttpUrl]
    previews: Previews
    images: Images
    num_downloads: int
    avg_rating: float
    num_ratings: int
    rate: Optional[HttpUrl]
    comments: Optional[HttpUrl]
    num_comments: int
    comment: Optional[HttpUrl]
    similar_sounds: Optional[HttpUrl]
    analysis: str
    analysis_frames: Optional[HttpUrl]
    analysis_stats: Optional[HttpUrl]
    is_explicit: bool


class OAuth2Payload(BaseModel):
    client_id: str
    client_secret: str
    grant_type: str
    code: str


class OAuth2Response(BaseModel):
    access_token: str
    expires_in: int
    refresh_token: str
    scope: str


class FreeSoundClient(Client):
    base_url: str = Field(default="https://freesound.org")

    async def token_endpoint(self, payload: OAuth2Payload) -> OAuth2Response:
        response = await self.post(
            "/apiv2/oauth2/access_token/",
            data=payload.dict(),
        )
        return OAuth2Response(**response)

    async def search_sound(
        self, query: str, filter: Filter, sort: Sort
    ) -> SearchResponse:
        response = await self.get(
            "/apiv2/search/text/",
            params=SearchQuery(
                query=query,
                filter=FilterMapping[filter](),
                sort=sort,
            ).dict(),
        )
        return SearchResponse(**response)

    async def get_sound(self, id: int) -> SoundInstance:
        response = await self.get(
            f"/apiv2/sounds/{id}/",
        )
        return SoundInstance(**response)

    async def download_sound(self, id: int) -> bytes:
        response = await self.get(
            f"/apiv2/sounds/{id}/download/",
        )
        return response

    async def get_me(self) -> dict[str, Any]:
        response = await self.get(
            "/apiv2/me/",
        )
        return response


def from_headers(headers: dict[str, str]) -> FreeSoundClient:
    return FreeSoundClient(headers=headers)
