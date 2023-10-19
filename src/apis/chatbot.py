from jinja2 import Template

from ..apis import *
from ..client import *
from ..schemas import *
from .freesound import *
from .storage import *


class ChatBot(Agent):
    prompt: str = Field(...)
    chat: ChatCompletion = Field(default_factory=ChatCompletion)
    instruction: Completion = Field(default_factory=Completion)
    vectors: VectorClient = Field(default_factory=VectorClient)
    embeddings: Embeddings = Field(default_factory=Embeddings)
    audio: Audio = Field(default_factory=Audio)
    image: Image = Field(default_factory=Image)
    storage: S3Client = Field(default_factory=S3Client)

    @property
    def template(self):
        return Template(self.prompt)

    async def run(self, text: str, **kwargs: Any):
        return await self.chat.run(text, self.template.render(**kwargs))
