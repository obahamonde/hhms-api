"""
Logging and error handling utilities for the Swagchain API.
"""
import asyncio
import functools
import logging
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import (Any, Callable, Coroutine, Generator, Literal, Sequence,
                    TypeAlias, TypeVar, cast)

import tiktoken
from aiohttp.web_exceptions import HTTPException
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import install
from rich.traceback import install as ins
from typing_extensions import ParamSpec

EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    TimeoutError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    UnicodeError,
    TypeError,
    ValueError,
    ZeroDivisionError,
    IndexError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    NotImplementedError,
    RecursionError,
    OverflowError,
    KeyError,
    Exception,
)
from dotenv import load_dotenv

load_dotenv()
TIKTOKEN_ENCODING_NAME = "cl100k_base"
Case: TypeAlias = Literal["snake", "pascal", "camel", "constant", "human"]
T = TypeVar("T")
P = ParamSpec("P")


def async_io(func: Callable[P, T]) -> Callable[P, Coroutine[T, Any, Any]]:
    """
    Decorator to convert an IO bound function to a coroutine by running it in a thread pool.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def async_cpu(func: Callable[P, T]) -> Callable[P, Coroutine[T, Any, Any]]:
    """
    Decorator to convert a CPU bound function to a coroutine by running it in a process pool.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with ProcessPoolExecutor() as pool:
            try:
                return await asyncio.get_running_loop().run_in_executor(
                    pool, func, *args, **kwargs
                )
            except RuntimeError:
                return await asyncio.get_event_loop().run_in_executor(
                    pool, func, *args, **kwargs
                )

    return wrapper


def setup_logging(name: str) -> logging.Logger:
    """
    Set's up logging using the Rich library for pretty and informative terminal logs.

    Arguments:
    name -- Name for the logger instance. It's best practice to use the name of the module where logger is defined.
    """
    install()
    ins()
    console = Console(record=True, force_terminal=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        tracebacks_extra_lines=2,
        tracebacks_theme="monokai",
        show_level=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[console_handler])
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.INFO)
    return logger_


logger = setup_logging(__name__)


def process_time(
    func: Callable[P, Coroutine[Any, Any, T] | T]
) -> Callable[P, Coroutine[Any, Any, T]]:
    """
    A decorator to measure the execution time of a coroutine.

    Arguments:
    func -- The coroutine whose execution time is to be measured.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        """
        Wrapper function to time the function call.
        """
        start = perf_counter()
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        end = perf_counter()
        logger.info(
            "Time taken to execute %s: %s seconds", wrapper.__name__, end - start
        )
        return result  # type: ignore

    return wrapper


def handle_errors(
    func: Callable[P, Coroutine[Any, Any, T] | T]
) -> Callable[P, Coroutine[Any, Any, T]]:
    """
    A decorator to handle errors in a coroutine.

    Arguments:
    func -- The coroutine whose errors are to be handled.
    """

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        """
        Wrapper function to handle errors in the function call.
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)  # type: ignore
        except EXCEPTIONS as exc:
            logger.error(exc.__class__.__name__)  # type: ignore
            logger.error(str(exc))
            raise HTTPException(reason=str(exc)) from exc

    return wrapper


def chunker(seq: Sequence[T], size: int) -> Generator[Sequence[T], None, None]:
    """
    A generator function that chunks a sequence into smaller sequences of the given size.

    Arguments:
    seq -- The sequence to be chunked.
    size -- The size of the chunks.
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def gen_emptystr() -> str:
    """
    A generator function that returns an empty string.
    """
    return cast(str, None)


def retry(
    tries: int = 3, delay: float = 1, factor: float = 2.0
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]
]:
    """
    A decorator to retry a coroutine if it fails, with exponential backoff.

    Arguments:

    tries -- The number of times to retry the coroutine.
    delay -- The initial delay between retries.
    factor -- The multiplier applied to the delay for each retry.
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """
            Wrapper function to retry the function call.
            """
            current_delay = delay
            for attempt in range(tries):
                try:
                    return await func(*args, **kwargs)
                except EXCEPTIONS as exc:
                    logger.error("Attempt %s/%s failed", attempt + 1, tries)
                    logger.error(str(exc))
                    if attempt + 1 == tries:
                        raise HTTPException(reason=str(exc)) from exc
                    else:
                        await asyncio.sleep(current_delay)
                        current_delay *= factor
            raise HTTPException(reason="All retries have been exhausted.")  # type: ignore

        return wrapper

    return decorator


def count_tokens(string: str, endcoding_name: str = TIKTOKEN_ENCODING_NAME) -> int:
    """Counts the number of tokens in a string."""
    encoding = tiktoken.get_encoding(endcoding_name)
    return len(encoding.encode(string))
