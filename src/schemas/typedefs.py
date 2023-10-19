from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from typing_extensions import ParamSpec

ImageModel: TypeAlias = Literal["dall-e"]
CompletionModel: TypeAlias = Literal["gpt-3.5-turbo-instruct", "davinci-002"]
ChatModel: TypeAlias = Literal["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
EmbeddingModel: TypeAlias = Literal["text-embedding-ada-002"]
AudioModel: TypeAlias = Literal["whisper-1"]
Model: TypeAlias = Union[
    ChatModel, EmbeddingModel, AudioModel, CompletionModel, ImageModel
]
Role: TypeAlias = Literal["user", "system", "assistant", "function"]
Size: TypeAlias = Literal["256x256", "512x512", "1024x1024"]
ImageFormat: TypeAlias = Literal["url", "base64"]
AudioFormat: TypeAlias = Literal["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
Vector: TypeAlias = List[float]
MetaDataValue: TypeAlias = Union[str, int, float, bool, List[str]]
MetaData: TypeAlias = Dict[str, MetaDataValue]

M = TypeVar("M", bound=Model)
P = ParamSpec("P")


class OpenAIResource(ABC, BaseModel):
    model: M  # type: ignore

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        ...


T = TypeVar("T")


class LazyProxy(Generic[T], ABC):
    def __init__(self) -> None:
        self.__proxied: T | None = None

    def __getattr__(self, attr: str) -> object:
        return getattr(self.__get_proxied__(), attr)

    def __repr__(self) -> str:
        return repr(self.__get_proxied__())

    def __dir__(self) -> Iterable[str]:
        return self.__get_proxied__().__dir__()

    def __get_proxied__(self) -> T:
        proxied = self.__proxied
        if proxied is not None:
            return proxied

        self.__proxied = proxied = self.__load__()
        return proxied

    def __set_proxied__(self, value: T) -> None:
        self.__proxied = value

    def __as_proxied__(self) -> T:
        return cast(T, self)

    @abstractmethod
    def __load__(self) -> T:
        ...


class Agent(ABC, BaseModel):
    name: str = Field(..., description="Name of the agent")

    @abstractmethod
    async def run(self, text: str, **kwargs: Any) -> Any:
        ...
