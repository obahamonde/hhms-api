"""HHMS `vector` module: A mini sdk of Pinecone's vector database API."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, TypeAlias, TypeVar, Union
from uuid import uuid4

from ..data import Document, Field  # pylint: disable=no-name-in-module
from .fauna_client import *

Vector: TypeAlias = List[Union[int, float]]
Value: TypeAlias = Union[int, float, str, bool, List[str]]
QueryJoin: TypeAlias = Literal["$and", "$or"]
QueryWhere: TypeAlias = Literal[
    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"
]
QueryKey: TypeAlias = Union[str, QueryWhere, QueryJoin]
QueryValue: TypeAlias = Union[Value, List[Value], "Query", List["Query"]]
Query: TypeAlias = Dict[QueryKey, QueryValue]
MetaData: TypeAlias = Dict[str, Value]


class QueryBuilder(object):
    def __init__(self, field: str = None, query: Query = None):  # type: ignore
        """
        Initializes a new QueryBuilder instance.

        Args:
            field (str, optional): The field to query. Defaults to None.
            query (Query, optional): The query dictionary. Defaults to an empty dictionary.
        """
        self.field = field
        self.query = query if query else {}

    def __repr__(self) -> str:
        """Returns the string representation of the query."""
        return f"{self.query}"

    def __call__(self, field_name: str) -> QueryBuilder:
        """
        Creates a new QueryBuilder instance with a given field name.

        Args:
            field_name (str): The field name to query.

        Returns:
            QueryBuilder: A new QueryBuilder instance.
        """
        return QueryBuilder(field_name)

    def __and__(self, other: QueryBuilder) -> QueryBuilder:
        """
        Combines two queries using the $and operator.

        Args:
            other (QueryBuilder): Another QueryBuilder instance.

        Returns:
            QueryBuilder: A new QueryBuilder instance with the combined query.
        """
        return QueryBuilder(query={"$and": [self.query, other.query]})

    def __or__(self, other: QueryBuilder) -> QueryBuilder:
        """
        Combines two queries using the $or operator.

        Args:
            other (QueryBuilder): Another QueryBuilder instance.

        Returns:
            QueryBuilder: A new QueryBuilder instance with the combined query.
        """
        return QueryBuilder(query={"$or": [self.query, other.query]})

    def __eq__(self, value: Value) -> QueryBuilder:  # type: ignore
        """
        Creates a new QueryBuilder instance with the $eq operator.

        Args:
            value (Value): The value to query.

        Returns:
            QueryBuilder: A new QueryBuilder instance with the combined query.

        """
        return QueryBuilder(query={self.field: {"$eq": value}})

    def __ne__(self, value: Value) -> QueryBuilder:  # type: ignore
        """
        Creates a new QueryBuilder instance with the $ne operator.

        Args:
            value (Value): The value to query.

        Returns:
            QueryBuilder: A new QueryBuilder instance with the combined query.

        """
        return QueryBuilder(query={self.field: {"$ne": value}})

    def __lt__(self, value: Value) -> QueryBuilder:
        """
        Creates a new QueryBuilder instance with the $lt operator.

        Args:

            value (Value): The value to query.

        Returns:

            QueryBuilder: A new QueryBuilder instance with the combined query.


        """

        return QueryBuilder(query={self.field: {"$lt": value}})

    def __le__(self, value: Value) -> QueryBuilder:
        """
        Creates a new QueryBuilder instance with the $lte operator.

        Args:

            value (Value): The value to query.

        Returns:

            QueryBuilder: A new QueryBuilder instance with the combined query.

        """

        return QueryBuilder(query={self.field: {"$lte": value}})

    def __gt__(self, value: Value) -> QueryBuilder:
        """

        Creates a new QueryBuilder instance with the $gt operator.

        Args:

            value (Value): The value to query.

        Returns:

            QueryBuilder: A new QueryBuilder instance with the combined query.

        """

        return QueryBuilder(query={self.field: {"$gt": value}})

    def __ge__(self, value: Value) -> QueryBuilder:
        """

        Creates a new QueryBuilder instance with the $gte operator.

        Args:

            value (Value): The value to query.

        Returns:

            QueryBuilder: A new QueryBuilder instance with the combined query.

        """

        return QueryBuilder(query={self.field: {"$gte": value}})

    def in_(self, values: List[Value]) -> QueryBuilder:
        """

        Creates a new QueryBuilder instance with the $in operator.

        Args:

            values (List[Value]): The values to query.

        Returns:

            QueryBuilder: A new QueryBuilder instance with the combined query.

        """

        return QueryBuilder(query={self.field: {"$in": values}})

    def nin_(self, values: List[Value]) -> QueryBuilder:
        """

        Creates a new QueryBuilder instance with the $nin operator.

        Args:

            values (List[Value]): The values to query.

        Returns:

            QueryBuilder: A new QueryBuilder instance with the combined query.

        """

        return QueryBuilder(query={self.field: {"$nin": values}})


class UpsertRequest(Document):
    """
    A request to upsert a vector.

    Args:
        id (str, optional): The ID of the vector. Defaults to a random UUID.
        values (Vector): The vector values.
        metadata (MetaData): The vector metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    values: Vector = Field(...)
    metadata: MetaData = Field(...)


class Embedding(Document):
    """

    A vector embedding.

    Args:

        values (Vector): The vector values.

        metadata (MetaData): The vector metadata.

    """

    values: Vector = Field(...)
    metadata: MetaData = Field(...)


class QueryRequest(Document):
    """
    A request to query vectors.

    Args:

        topK (int, optional): The number of results to return. Defaults to 10.

        filter (Query): The query filter.

        includeMetadata (bool, optional): Whether to include metadata in the response. Defaults to True.

        vector (Vector): The vector to query.

    """

    topK: int = Field(default=10)
    filter: Dict[str, Any] = Field(...)
    includeMetadata: bool = Field(default=True)
    vector: Vector = Field(...)


class QueryMatch(Document):
    """
    A query match.

    Args:

        id (str): The ID of the vector.

        score (float): The match score.

        metadata (MetaData): The vector metadata.

    """

    id: str = Field(...)
    score: float = Field(...)
    metadata: MetaData = Field(...)


class QueryResponse(Document):
    """

    A response to a query request.

    Args:

        matches (List[QueryMatch]): The query matches.

    """

    matches: List[QueryMatch] = Field(...)


class UpsertResponse(Document):
    """

    A response to an upsert request.

    Args:

        upsertedCount (int): The number of vectors upserted.

    """

    upsertedCount: int = Field(...)


class VectorClient(Client):
    """

    A client for the vector database.

    Args:

        api_key (str): The API key.

        api_endpoint (str): The API endpoint.

    """

    base_url: str = Field(
        default_factory=lambda: os.getenv("PINECONE_API_URL", "https://api.pinecone.io")
    )
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "api-key": os.getenv("PINECONE_API_KEY"),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )

    async def upsert(self, embeddings: List[Embedding]) -> UpsertResponse:
        """

        Upserts a list of vector embeddings.

        Args:

            embeddings (List[Embedding]): The vector embeddings.

        Returns:

            UpsertResponse: The upsert response.

        """
        return await self.post(
            "/vectors/upsert",
            json={
                "vectors": [
                    UpsertRequest(
                        values=embedding.values, metadata=embedding.metadata
                    ).dict()
                    for embedding in embeddings
                ]
            },
        )

    async def query(
        self, expr: Query, vector: Vector, topK: int, includeMetadata: bool = True
    ) -> QueryResponse:
        """

        Queries the vector database.

        Args:

            expr (Query): The query expression.

            vector (Vector): The vector to query.

            topK (int): The number of results to return.

            includeMetadata (bool, optional): Whether to include metadata in the response. Defaults to True.

        Returns:

            QueryResponse: The query response.

        """

        payload = QueryRequest(
            topK=topK,
            filter=expr,  # type: ignore
            vector=vector,
            includeMetadata=includeMetadata,
        ).dict()
        response = await self.post(
            "/query",
            json=payload,
        )
        return QueryResponse(**response)
