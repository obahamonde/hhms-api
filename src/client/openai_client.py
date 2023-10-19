from tempfile import NamedTemporaryFile
from typing import Any, Set, Type

import openai
from pydantic import Field, root_validator  # pylint: disable=no-name-in-module

from ..schemas import AudioFormat  # pylint: disable=no-name-in-module
from ..schemas import (AudioModel, BaseModel, ChatModel, CompletionModel, Dict,
                       EmbeddingModel, ImageFormat, ImageModel, List,
                       OpenAIResource, Role, Size, Vector)
from ..utils import count_tokens, retry
from .pinecone_client import *


class CompletionRequest(BaseModel):
    model: CompletionModel = Field(default="gpt-3.5-turbo-instruct")
    prompt: str = Field(...)
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=1024)
    stream: bool = Field(default=False)


class CompletionChoice(BaseModel):
    index: int = Field(...)
    finish_reason: str = Field(...)
    text: str = Field(...)


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(...)
    completion_tokens: int = Field(...)
    total_tokens: int = Field(...)


class CompletionResponse(BaseModel):
    id: str = Field(...)
    object: str = Field(...)
    created: int = Field(...)
    model: CompletionModel = Field(...)
    choices: List[CompletionChoice] = Field(...)
    usage: CompletionUsage = Field(...)


class Message(BaseModel):
    """
    Represents a message within the chatcompletion API pipeline.
    """

    role: Role = Field(default="user")
    content: str = Field(...)


class ChatCompletionRequest(BaseModel):
    model: ChatModel = Field(default="gpt-3.5-turbo-16k")
    messages: List[Message] = Field(...)
    temperature: float = Field(default=0.5)
    max_tokens: int = Field(default=4096)
    stream: bool = Field(default=False)


class ChatCompletionUsage(BaseModel):
    """Token usage statistics for a chat completion API call."""

    prompt_tokens: int = Field(...)
    completion_tokens: int = Field(...)
    total_tokens: int = Field(...)


class ChatCompletionChoice(BaseModel):
    index: int = Field(...)
    message: Message = Field(...)
    finish_reason: str = Field(...)


class ChatCompletionResponse(BaseModel):
    id: str = Field(...)
    object: str = Field(...)
    created: int = Field(...)
    model: str = Field(...)
    choices: List[ChatCompletionChoice] = Field(...)
    usage: ChatCompletionUsage = Field(...)


class EmbeddingUssage(BaseModel):
    """Token usage statistics for an embedding API call."""

    prompt_tokens: int = Field(...)
    total_tokens: int = Field(...)


class CreateImageResponse(BaseModel):
    created: float = Field(...)
    data: List[Dict[ImageFormat, str]] = Field(...)


class CreateImageRequest(BaseModel):
    """Request to create an image from a prompt. Use default values for configuration unless specified."""

    prompt: str = Field(...)
    n: int = Field(default=1)
    size: Size = Field(default="1024x1024")
    response_format: ImageFormat = Field(default="url")


class FineTuneSample(BaseModel):
    messages: List[Message] = Field(..., max_items=3, min_items=2)

    @root_validator
    @classmethod
    def check_messages(cls: Type[BaseModel], values: Dict[str, Any]):
        roles: Set[Role] = set()
        for message in values["messages"]:
            roles.add(message.role)
        assert len(roles) == len(
            values["messages"]
        ), "All messages must be from different roles."
        return values


class FineTuneRequest(BaseModel):
    __root__: List[FineTuneSample] = Field(..., min_items=10, max_items=100000)

    def __call__(self):
        with NamedTemporaryFile("w", suffix=".json") as f:
            data = self.json()
            assert count_tokens(data) < 4096, "Data too large."
            f.write(data)
            f.flush()
            return f


class AudioRequest(BaseModel):
    file: bytes = Field(...)
    format: AudioFormat = Field(default="mp3")

    def __call__(self):
        with NamedTemporaryFile("wb", suffix=f".{self.format}") as f:
            f.write(self.file)
            f.flush()
            assert len(f.read()) < 25 * 1024 * 1024, "File too large."
            return f


class ChatCompletion(OpenAIResource):
    """OpenAI Chat Completion API."""

    model: ChatModel = Field(default="gpt-3.5-turbo-16k")

    @retry()
    async def run(self, text: str, context: str):  # type: ignore
        request = ChatCompletionRequest(
            messages=[Message(content=text), Message(content=context, role="system")]
        )
        response = await openai.ChatCompletion.acreate(**request.dict())  # type: ignore
        return ChatCompletionResponse(**response)  # type: ignore

    async def stream(self, text: str, context: str) -> AsyncGenerator[str, None]:
        request = ChatCompletionRequest(
            messages=[Message(content=text), Message(content=context, role="system")],
            stream=True,
        )
        response = await openai.ChatCompletion.acreate(**request.dict())  # type: ignore
        async for message in response:  # type: ignore
            data = message.choices[0].delta.get("content", None)  # type: ignore
            yield data  # type: ignore


class Completion(OpenAIResource):
    """OpenAI Completion API."""

    model: CompletionModel = Field(default="gpt-3.5-turbo-instruct")

    @retry()
    async def run(self, text: str):  # type: ignore
        request = CompletionRequest(prompt=text)
        response = await openai.Completion.acreate(**request.dict())  # type: ignore
        return CompletionResponse(**response)  # type: ignore

    async def stream(self, text: str) -> AsyncGenerator[str, None]:
        request = CompletionRequest(prompt=text, stream=True)
        response = await openai.Completion.acreate(**request.dict())  # type: ignore
        async for message in response:  # type: ignore
            data = message.choices[0].get("text", None)  # type: ignore
            yield data  # type: ignore


class Embeddings(OpenAIResource):
    """OpenAI Embeddings API."""

    model: EmbeddingModel = Field(default="text-embedding-ada-002")

    @retry()
    async def run(self, texts: List[str]) -> List[Vector]:  # type: ignore
        response = await openai.Embedding.acreate(input=texts, model=self.model)  # type: ignore
        return [r.embedding for r in response.data]  # type: ignore


class Image(OpenAIResource):
    """OpenAI Image API."""

    model: ImageModel = Field(default="dall-e")
    size: Size = Field(default="1024x1024")
    format: ImageFormat = Field(default="url")

    @retry()
    async def run(self, text: str, n: int = 1) -> List[str]:  # type: ignore
        response = await openai.Image.acreate(prompt=text, n=n, size=self.size, response_format=self.format)  # type: ignore
        return [r[self.format] for r in response.data]  # type: ignore


class Audio(OpenAIResource):
    """OpenAI Audio API."""

    model: AudioModel = Field(default="whisper-1")

    @retry()
    async def run(self, content: bytes, audioformat: AudioFormat = "wav") -> str:  # type: ignore
        response = await openai.Audio.acreate(self.model, AudioRequest(file=content, format=audioformat)())  # type: ignore
        return response.get("text", "")  # type: ignore
