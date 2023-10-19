"""
Exceptions for aiofauna
"""

from json import JSONDecodeError

from aiohttp import (
    ClientConnectionError,
    ClientConnectorError,
    ClientConnectorSSLError,
    ClientError,
    ContentTypeError,
    ServerTimeoutError,
    WSServerHandshakeError,
)
from aiohttp.web_exceptions import HTTPException

from ..faunadb.errors import FaunaError, FaunaException


class AiofaunaException(Exception):
    """
    Base exception.
    """

    def __init__(self, message: str = None, *args: object) -> None:  # type: ignore
        if message is None:  # type: ignore
            message = self.__doc__  # type: ignore
        super().__init__(message, *args)


EXCEPTIONS = (
    # aiohttp exceptions
    ClientConnectionError,
    ClientConnectorError,
    ClientConnectorSSLError,
    ClientError,
    ContentTypeError,
    ServerTimeoutError,
    WSServerHandshakeError,
    # aiofauna exceptions
    AiofaunaException,
    # faunadb exceptions
    FaunaError,
    FaunaException,
    # json exceptions
    JSONDecodeError,
    # aiohttp.web_exceptions exceptions
    HTTPException,
    ValueError,
    KeyError,
    TypeError,
    Exception,
    UnicodeError,
    RuntimeError,
)
