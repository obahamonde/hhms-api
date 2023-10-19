import os

from boto3 import Session
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

from ..schemas import LazyProxy
from ..utils import async_io  # type: ignore


class S3Client(BaseModel):
    bucket: str = Field(default=os.environ["AWS_S3_BUCKET"])

    @property
    def client(self):
        return Session().client(service_name="s3", region_name="us-east-1")  # type: ignore

    @async_io
    def put_object(self, key: str, body: bytes):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": os.environ["AWS_S3_BUCKET"], "Key": key},
        )

    @async_io
    def get_object(self, key: str):
        return self.client.generate_presigned_url(
            ClientMethod="get_object", Params={"Bucket": self.bucket, "Key": key}
        )

    @async_io
    def delete_object(self, key: str):
        return self.client.delete_object(Bucket=self.bucket, Key=key)